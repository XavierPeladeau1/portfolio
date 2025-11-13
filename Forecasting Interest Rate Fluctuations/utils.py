from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import (
    Diff,
    InvertibleMapper,
    MissingValuesFiller,
    Scaler,
)
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.callbacks import Callback
import torch
from fredapi import Fred


load_dotenv()

#### Deep Learning utils
class LossAccumulatorCallback(Callback):
    """
    PyTorch Lightning Callback to accumulate training and validation losses per epoch.
    
    Attributes:
        train_losses (list): List to store training loss for each epoch.
        val_losses (list): List to store validation loss for each epoch.
    """
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        
        super().__init__()
    
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called when the train epoch ends.
        
        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The model being trained.
        """
        # Retrieve the training loss from logged metrics
        # Assumes that 'train_loss' is the key used for logging training loss
        train_loss = trainer.callback_metrics.get('train_loss')
        
        if train_loss is not None:
            # Detach and move to CPU if necessary
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.detach().cpu().item()
            self.train_losses.append(train_loss)
            # print(f"Accumulated train_loss: {self.train_losses}")
        else:
            print("train_loss not found in callback_metrics.")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss')
        
        if val_loss is not None:
            # Detach and move to CPU if necessary
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.detach().cpu().item()
            self.val_losses.append(val_loss)
        else:
            print("train_loss not found in callback_metrics.")
    

    def on_train_end(self, trainer, pl_module):
        """
        Called when the training ends.
        
        Args:
            trainer (pl.Trainer): The trainer instance.
            pl_module (pl.LightningModule): The model that was trained.
        """
        pass

def save_results(hparams, eval_metrics, output_path):
    output_path = Path(output_path)
    output_path = "results/" + output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    include_header = not output_path.exists()
    results = pd.concat([pd.Series(hparams), eval_metrics.mean()])
    pd.DataFrame(results).T.to_csv(
        output_path, mode="a", header=include_header, index=False
    )

def make_forecasts(model, ts: TimeSeries, ts_scaled: TimeSeries, covariates_scaled: TimeSeries, pipeline:Pipeline) -> pd.DataFrame:
    forecasts = pd.DataFrame()

    val_df_scaled = ts_scaled.drop_before(pd.Timestamp("2012-12-31")).pd_dataframe()


    # Make forecasts for each date in the validation set
    for t in val_df_scaled.index:

        # Get data up to date t
        ts_up_to_t = ts_scaled.drop_after(t)
        if covariates_scaled is not None:
            covariates = covariates_scaled.drop_after(t)
        else:
            covariates = None

        # print(f"Producing forecasts made at date: {ts_up_to_t.end_time()}")
        if model.supports_probabilistic_prediction:
            # Make forecasts
            pred = model.predict(n=36, series=ts_up_to_t, past_covariates=covariates, num_samples=500, verbose=False)

            # Get values for each quantile and unscale
            pred_quantiles_unscaled = {q: unscale_series(pred.quantile(q), pipeline, ts_scaled).pd_series() for q in [0.05, 0.1, 0.5, 0.9, 0.95]}
            # print(pred_unscaled)
            # print(pred)
        else:
            pred = model.predict(n=36, series=ts_up_to_t, past_covariates=covariates)
            pred_unscaled = unscale_series(pred, pipeline, ts_scaled).pd_series()
            pred_quantiles_unscaled = {0.5: pred_unscaled}
            for q in [0.05, 0.1, 0.9, 0.95]:
                pred_quantiles_unscaled[q] = pred_unscaled

        # Get labels (real values) for the period
        three_years = t + (pd.Timedelta(days=364)*3)
        labels = ts.pd_dataframe().loc[t:three_years]

        # If labels don't extend to the end of the forecast period, create NA values for the missing dates
        if labels.index[-1] < three_years:
            missing_dates = pd.date_range(labels.index[-1] + pd.offsets.MonthEnd(), three_years, freq="ME")
            missing_df = pd.DataFrame(index=missing_dates, columns=labels.columns)
            labels = pd.concat([labels, missing_df])

        # print(labels)



        # Create a dataframe with the forecasted values
        labels["lowest_q"] = pred_quantiles_unscaled[0.05]
        labels["low_q"] = pred_quantiles_unscaled[0.1]
        labels["forecast"] = pred_quantiles_unscaled[0.5]
        labels["high_q"] = pred_quantiles_unscaled[0.9]
        labels["highest_q"] = pred_quantiles_unscaled[0.95]
        labels["error"] = labels["US_TB_YIELD_10YRS"] - labels["forecast"]
        labels["forecast_date"] = ts_up_to_t.end_time()

        # print(labels)
        # Append to forecasts
        forecasts = pd.concat([forecasts, labels])


    # Horizon is the number of months between the forecast date and the date of the forecast
    forecasts["horizon"] = (forecasts.index.to_period("M") - forecasts.forecast_date.dt.to_period("M")).map(lambda x: x.n)

    # Print evaluation metrics
    # print(forecasts.groupby(by="horizon").mean()[["error"]])

    return forecasts

def get_ts_by_forecast_horizon(pred_df):
    forecast_by_horizon = {}
    for h in pred_df["horizon"].unique():
        fore = pred_df[pred_df["horizon"] == h]
        fore = fore.asfreq("ME")
        # forecast_by_horizon[h] = TimeSeries.from_dataframe(fore, value_cols=["forecast"])
        values = np.stack([fore["lowest_q"], fore["low_q"], fore["forecast"], fore["high_q"], fore["highest_q"]], axis=1)
        values = np.expand_dims(values, axis=1)
        forecast_by_horizon[h] = TimeSeries.from_times_and_values(times=fore.index, values=values)

    return forecast_by_horizon

def evaluate_by_horizon(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate forecasts by horizon
    Args:
        forecasts_df: DataFrame containing forecasts and real values. Must have columns "forecast", "US_TB_YIELD_10YRS" and "horizon".
        US_TB_YIELD_10YRS is the real value. forecast is the forecasted value. horizon is the number of months between the forecast date and the date of the forecast
    """
    # Compute error and squared error
    forecasts_df["abs_pct_error"] = ((forecasts_df["US_TB_YIELD_10YRS"] - forecasts_df["forecast"]) / forecasts_df["US_TB_YIELD_10YRS"]).abs() * 100
    forecasts_df["squared_error"] = forecasts_df["error"]**2
    forecasts_df["abs_error"] = forecasts_df["error"].abs()

    # Group by horizon and compute mean error and mean squared error
    grouped = forecasts_df.groupby(by="horizon").mean()[["abs_pct_error", "squared_error", "abs_error"]].add_prefix("mean_")
    grouped["root_mean_squared_error"] = grouped["mean_squared_error"]**0.5

    return grouped


#### Plot utils
def plot_training_history(train_losses, val_losses):
    # Create the figure
    fig = go.Figure()

    # Add traces for training and validation loss
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(train_losses) + 1)),
            y=train_losses,
            name="Training Loss",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(val_losses) + 1)),
            y=val_losses,
            name="Validation Loss",
            line=dict(color="red"),
        )
    )

    # Update layout
    fig.update_layout(
        title="Training and Validation Loss Over Time",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        template="plotly_white",
    )

    # Show the plot
    fig.show()


def plot_forecast_for_horizon(h, forecasts_ts, axs, ts):
    figsize = (9, 6)
    lowest_q, low_q, high_q, highest_q = 0.05, 0.1, 0.9, 0.95
    label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
    label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

    fcast = forecasts_ts[h]

    # plot actual series
    # plt.figure(figsize=figsize)
    ts[fcast.start_time() :].plot(label="actual", ax=axs)

    # plot prediction with quantile ranges
    fcast.plot(low_quantile=0.1, high_quantile=0.95, label=label_q_outer, ax=axs)
    fcast.plot(low_quantile=0.3, high_quantile=0.7, label=label_q_inner, ax=axs)

    # if axs.get_legend():
    #     axs.get_legend().remove()
    # axs.set_xlabel("")

    # plt.title(f"MAPE: {mape(ts, fcast):.2f}%")
    axs.set_title(f"{h}-month ahead forecast", fontsize=8)
    plt.legend()
    plt.show()


### Data Loading and Preprocessing
def fetch_fred_data(start_date="1960-01-01", end_date=None):
    """
    Fetch economic data from FRED API.

    Args:
        start_date: Start date for data retrieval (default: "1960-01-01")
        end_date: End date for data retrieval (default: None, which fetches all available data)

    Returns:
        DataFrame with economic indicators

    Note: Requires FRED API key in environment variable FRED_API_KEY
    """
    # Get API key from environment variable
    api_key = os.environ.get('FRED_API_KEY')
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not found in environment variables. "
            "Get your free API key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and set it with: export FRED_API_KEY='your_api_key'"
        )

    fred = Fred(api_key=api_key)

    # Map column names to FRED series IDs
    series_mapping = {
        'FFED': 'DFF',  # Federal Funds Effective Rate
        'US_PERSONAL_SPENDING_PCE': 'PCEPI',  # Personal Consumption Expenditures: Chain-type Price Index
        'US_CPI': 'CPIAUCSL',  # Consumer Price Index for All Urban Consumers: All Items
        'US_TB_YIELD_10YRS': 'DGS10',  # 10-Year Treasury Constant Maturity Rate
        'US_TB_YIELD_1YR': 'DGS1',  # 1-Year Treasury Constant Maturity Rate
        'US_TB_YIELD_2YRS': 'DGS2',  # 2-Year Treasury Constant Maturity Rate
        'US_TB_YIELD_3YRS': 'DGS3',  # 3-Year Treasury Constant Maturity Rate
        'US_TB_YIELD_5YRS': 'DGS5',  # 5-Year Treasury Constant Maturity Rate
        'US_TB_YIELD_3MTHS': 'DGS3MO',  # 3-Month Treasury Constant Maturity Rate
        'US_UNEMPLOYMENT_RATE': 'UNRATE',  # Unemployment Rate
        'SNP_500': 'SP500',  # S&P 500 Index
    }

    # Fetch each series
    data = {}
    for col_name, series_id in series_mapping.items():
        try:
            series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            data[col_name] = series
        except Exception as e:
            print(f"Warning: Failed to fetch {col_name} ({series_id}): {e}")
            data[col_name] = pd.Series(dtype=float)

    # Combine into DataFrame
    df = pd.DataFrame(data)
    df.index.name = 'DATE'

    return df


def lag_monthly_macro_variables(df):
    """
    Macro-economic metrics such as CPI, PCE and unemployment rate are only known 1 month after the period they cover.
    Example: On January 15th 2016, we want to produce forecasts for February 15th 2016, but
    we only have CPI data up to December 2015.

    For this reason, we lag historical macro-economic variables by 1 prior to modeling
    """
    # print("Before shift:")
    # print(df.head())
    for col in ["US_CPI", "US_UNEMPLOYMENT_RATE", "US_PERSONAL_SPENDING_PCE"]:
        df[col] = df[col].shift(1)

    # print("After shift:")
    # print(df.head())

    return df


def load_jorge_data():
    # deux_semaines = pd.read_csv("data/excel_jorge/Variables_Chomage_US_2semaines.csv")
    # deux_semaines["DATE"] = pd.to_datetime(deux_semaines["DATE"])

    trimestrielles = pd.read_excel("data/excel_jorge/Variables_US_Trimestrielles.xlsx")
    trimestrielles = trimestrielles.replace("Nan", pd.NA).replace("<NA>", pd.NA)
    trimestrielles["DATE"] = pd.to_datetime(trimestrielles["DATE"])
    trimestrielles["W068RCQ027SBEA"] = pd.to_numeric(trimestrielles["W068RCQ027SBEA"])
    trimestrielles = trimestrielles.set_index("DATE").asfreq("QE", method="ffill")
    trimestrielles = trimestrielles[["GDPC1", "GPDI", "W068RCQ027SBEA"]]
    trimestrielles = trimestrielles.resample("ME").interpolate()


    weekly = pd.read_csv("data/excel_jorge/Variables_US_Weekly.csv")
    weekly = weekly.replace("Nan", pd.NA).replace("<NA>", pd.NA)
    weekly["DATE"] = pd.to_datetime(weekly["DATE"])
    weekly = weekly.set_index("DATE").asfreq("W", method="ffill").resample("ME").mean()
    weekly = weekly[["TOTBKCR"]]

    autres = pd.read_csv("data/excel_jorge/Variables_US.csv")
    autres["DATE"] = pd.to_datetime(autres["DATE"])
    autres = autres.set_index("DATE").asfreq("MS")
    autres = autres.replace("Nan", pd.NA)
    autres = autres.apply(pd.to_numeric, errors="ignore")
    autres = autres.resample("ME").mean()

    autres = autres[
        ["STICKCPIM157SFRBATL", "MICH", "AWHMAN", "EMRATIO", "STDSL", "EXPINF10YR", "PAYEMS"]
    ]

    # Merge all data
    jorge_df = trimestrielles.merge(weekly, left_index=True, right_index=True, how="left")
    jorge_df = jorge_df.merge(autres, left_index=True, right_index=True, how="left")

    return jorge_df


def load_data(test=False):
    """Load raw data and construct DataFrame with all **unscaled** features"""

    # Load SF FED data
    sf_df = pd.read_excel(
        Path(__file__).parent / "data/sf_fed/news_sentiment_data.xlsx",
        sheet_name="Data",
    )
    sf_df = sf_df.set_index("date").asfreq("B").resample("ME").mean()
    sf_df = sf_df.rolling(window=12).mean().dropna()  # Smooth data

    # Fetch macro-economic data from FRED API
    df = fetch_fred_data(start_date="1960-01-01")

    # Set date index and frequency
    df = df.reset_index()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").asfreq("B")

    # Compute yield curve indicator
    df["YIELD_CURVE"] = df["US_TB_YIELD_10YRS"] - df["US_TB_YIELD_3MTHS"]

    # Resample to monthly frequency
    df = df.resample("ME").mean()

    if not test:
        # Keep last year for testing
        df = df[df.index <= "2023-08-31"]



    ###### Add Jorge's data
    jorge_df = load_jorge_data()
    df = df.merge(jorge_df, left_index=True, right_index=True, how="left")


    # Merge with SF FED data
    df = df.merge(sf_df, left_index=True, right_index=True, how="left").rename(
        columns={"News Sentiment": "NEWS_SENTIMENT"}
    )

    # Keep only data from 1980 onwards
    df = df[df.index >= "1980-01-01"]

    # Lag macro-economic variables
    df = lag_monthly_macro_variables(df)

    df = df.astype(np.float32)

    return df


def scale_ts(series, should_diff, diff_order=1, should_scale=True, should_log=True):
    """Scale TimeSeries and apply transformations"""
    log_transformer = InvertibleMapper(fn=np.log1p, inverse_fn=np.expm1, name="log1p")
    scaler = Scaler(StandardScaler())
    filler = MissingValuesFiller()
    differentiator = Diff(dropna=True, lags=diff_order)

    operations = [filler]
    if should_diff:
        operations.append(differentiator)
    # if should_log:
    #     operations.append(log_transformer)
    if should_scale:
        operations.append(scaler)

    pipeline = Pipeline(operations)
    series_scaled = pipeline.fit_transform(series)

    return pipeline, series_scaled


def unscale_series(series: TimeSeries, pipeline: Pipeline, ts_scaled):
    series_start_time = series.start_time()
    full_history = ts_scaled.drop_after(series_start_time).append(series)

    unscaled_full = pipeline.inverse_transform(full_history, partial=True)

    idx_start_time = unscaled_full.get_index_at_point(series_start_time)
    unscaled = unscaled_full.drop_before(idx_start_time - 1)

    return unscaled


def df2ts(df):
    # Create a TimeSeries object
    ts = TimeSeries.from_dataframe(df, value_cols=["US_TB_YIELD_10YRS"])

    # Create covariates that will be differenced
    covars_diff = df[
        [
            "FFED",
            "US_TB_YIELD_1YR",
            "US_TB_YIELD_2YRS",
            "US_TB_YIELD_3YRS",
            "US_TB_YIELD_5YRS",
            "US_TB_YIELD_3MTHS",
            "US_PERSONAL_SPENDING_PCE",
            "EXPINF10YR",  # EXPECTED INFLATION 10 YR
            "AWHMAN",
            "STDSL",  # SMALL DEPOSITS
            "GDPC1",  # GDP
            "GPDI",
            "W068RCQ027SBEA",
            "TOTBKCR",
            "US_UNEMPLOYMENT_RATE",
        ]
    ]
    covars_diff = TimeSeries.from_dataframe(covars_diff)

    covars_diff_yoy = df[["SNP_500", "US_CPI"]]  # STOCK MARKET  # INFLATION
    covars_diff_yoy = TimeSeries.from_dataframe(covars_diff_yoy)

    # Create covariates that will not be differenced
    covars_nodiff = df[
        [
            "NEWS_SENTIMENT",
            "YIELD_CURVE",
            "STDSL",
            "STICKCPIM157SFRBATL",  # STICKY CPI PCT CHANGE
            "MICH",  # EXPECTED PCT INFLATION 1 YR
            "EMRATIO"
        ]
    ]
    covars_nodiff = TimeSeries.from_dataframe(covars_nodiff)

    return ts, covars_diff, covars_diff_yoy, covars_nodiff
