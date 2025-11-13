"""
SARIMA Model for Electricity Consumption Forecasting

This script trains SARIMA (Seasonal AutoRegressive Integrated Moving Average) models
to forecast monthly electricity consumption for different MRCs (regional municipalities)
and economic sectors in Quebec.

The workflow consists of:
1. Training models on all available MRC-sector combinations (2016-2021)
2. Validating and selecting optimal hyperparameters using 2022 data
3. Generating final predictions for filtered MRCs on 2023 test data
"""

import logging
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import patsy
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

from utils import load_and_prepare_dataset

# Setup logging and warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()
logger.addHandler(logging.FileHandler("arima.log", mode="w"))


def load_data():
    """Load raw electricity consumption data from Excel file."""
    df = pd.read_excel("data/consommation-historique-mrc-11mars2024.xlsx")
    df = df.dropna(subset=["MRC_TXT"])
    df = df.set_index(df["ANNEE_MOIS"].infer_objects()).sort_index()
    df = df.rename(columns={"Total (kWh)": "total_kwh"})

    return df


def get_consumption_for(df: pd.DataFrame, mrc, sector):
    """Filter dataset for a specific MRC and sector, and prepare time index."""
    df = df[(df["mrc"] == mrc) & (df["sector"] == sector)]
    df = df.set_index(pd.to_datetime(df["date"], format="%Y-%m-%d")).sort_index()
    df["month"] = df.index.month.astype(str)
    return df


def acf_pacf(series: pd.Series):
    """Plot ACF and PACF for seasonally differenced series to identify SARIMA parameters."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    fig = sm.graphics.tsa.plot_acf(series.diff(12).dropna(), lags=40, ax=axes[0])
    fig = sm.graphics.tsa.plot_pacf(series.diff(12).dropna(), lags=40, ax=axes[1])

    plt.show()


def test_models(df, formula):
    """
    Perform grid search over SARIMA hyperparameters to find optimal p and q.

    Tests all combinations of p and q in range [0, 2] and returns the pair
    with the lowest AIC score.
    """
    results = []
    y, X = patsy.dmatrices(formula, data=df, return_type="dataframe")

    # Grid search over p and q parameters
    for p in range(3):
        for q in range(3):
            if p == 0 and q == 0:
                continue

            model = sm.tsa.statespace.SARIMAX(
                endog=y,
                exog=X,
                order=(p, 1, q),
                seasonal_order=(p, 1, q, 12),
            )
            res = model.fit(disp=False)
            results.append([p, q, res.aic])

    # Select best parameters based on AIC
    best_p, best_q = (
        pd.DataFrame(results, columns=["p", "q", "aic"])
        .set_index(["p", "q"])
        .idxmin()
        .at["aic"]
    )

    return best_p, best_q


def fit_model(train_df, p, q, formula):
    """Fit a SARIMA model with given hyperparameters and formula."""
    y, X = patsy.dmatrices(formula, data=train_df, return_type="dataframe")

    model = sm.tsa.statespace.SARIMAX(
        endog=y,
        exog=X,
        order=(p, 1, q),
        seasonal_order=(p, 1, q, 12),
    )
    res = model.fit(disp=False)

    return res


def plot_residuals(res):
    """Plot model residuals for diagnostics."""
    fig = px.line(
        res.resid[12:],
        title="Residuals",
    )
    fig.show()


def forecast(model, test_df, formula):
    """
    Generate forecasts and compute error metrics.

    Returns forecast, RMSE, MAPE, and actual values.
    """
    y, X = patsy.dmatrices(formula, data=test_df, return_type="dataframe")

    forecast = model.forecast(steps=12, exog=X)
    forecast.index = y.index

    # Calculate error metrics
    rmse = (forecast.sub(y["total_kwh"])**2).mean()**0.5
    mape = (forecast.sub(y["total_kwh"]).abs() / y["total_kwh"]).mean()

    return forecast, rmse, mape, y


def plot_predictions(test_df, fore):
    """Plot forecast with confidence intervals against actual data."""
    import plotly.graph_objects as go

    fore = fore.summary_frame().drop(columns=["mean_se"])

    fig = go.Figure()

    # Add actual data
    fig.add_trace(
        go.Scatter(x=test_df.index, y=test_df["total_kwh"], mode="lines", name="Actual")
    )

    # Add forecasted mean
    fig.add_trace(
        go.Scatter(x=fore.index, y=fore["mean"], mode="lines", name="Forecast")
    )

    # Add confidence intervals as shaded areas
    fig.add_trace(
        go.Scatter(
            x=fore.index.tolist() + fore.index[::-1].tolist(),
            y=fore["mean_ci_upper"].tolist() + fore["mean_ci_lower"][::-1].tolist(),
            fill="toself",
            fillcolor="rgba(0,100,80,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
            name="Confidence Interval",
        )
    )

    fig.update_layout(
        title="Forecast with Confidence Intervals",
        xaxis_title="Date",
        yaxis_title="Total kWh",
    )

    return fig

if __name__ == "__main__":

    # Define MRC filter for predictions - modify this list to change which MRCs to generate predictions for
    MRC_FILTER = ["Drummond", "Les Etchemins"]

    # Define model formulas (exogenous variables)
    formulas = {
        "mean_temp__month": "total_kwh ~ tavg + month",
    }

    # Load all data (no filter) for training
    full_df = load_and_prepare_dataset()

    # Initialize results dataframe
    results_df = pd.DataFrame()
    results_df[["MRC", "SECTOR"]] = full_df[["mrc", "sector"]].drop_duplicates()
    models = {}

    # Phase 1: Train and validate models for all MRC-sector combinations
    for results_col, formula in formulas.items():
        for mrc in full_df["mrc"].unique():
            if mrc not in MRC_FILTER:
                continue
            for sector in full_df["sector"].unique():
                # Split data into train/validation/test sets
                df = get_consumption_for(full_df, mrc, sector)
                train_df = df["2016":"2021"]
                validation_df = df["2022":"2022"]
                test_df = df["2023":]

                try:
                    # Find optimal hyperparameters
                    best_p, best_q = test_models(train_df, formula=formula)

                    # Fit model with best parameters
                    model = fit_model(train_df, best_p, best_q, formula=formula)

                    # Evaluate on validation set
                    fore, rmse, mape, y = forecast(model, validation_df, formula)

                    # Log results
                    logger.info(f"MRC: {mrc}, SECTOR: {sector},  RMSE: {rmse}, MAPE: {mape}")
                    logger.info(f"Best p: {best_p}, Best q: {best_q}")

                    # Store metrics
                    results_df.loc[((results_df["MRC"] == mrc) & (results_df["SECTOR"] == sector)), results_col+"_rmse"] = rmse
                    results_df.loc[((results_df["MRC"] == mrc) & (results_df["SECTOR"] == sector)), results_col+"_mape"] = mape

                    # Save model for later prediction
                    models[(mrc, sector)] = [model, best_p, best_q]

                except Exception as e:
                    logger.error(f"Error for MRC: {mrc}, {sector}, {e}")
                    continue

    # Phase 2: Generate final predictions for filtered MRCs on test set (2023)
    for (mrc, sector), (model, best_p, best_q) in models.items():
        # Skip if not in filter
        if mrc not in MRC_FILTER:
            continue

        # Retrain on extended dataset (include validation year)
        df = get_consumption_for(full_df, mrc, sector)
        train_df = df["2016":"2022"]
        test_df = df["2023":]

        formula = formulas["mean_temp__month"]

        # Refit model with extended training data
        model = fit_model(train_df, best_p, best_q, formula=formula)

        # Generate predictions for 2023
        fore, rmse, mape, y = forecast(model, test_df, formula)

        # Visualize results
        fig = px.line(fore.rename("Prédiction"), title=f"{mrc} - {sector}")
        fig.add_trace(go.Scatter(x=y.index, y=y["total_kwh"], mode="lines", name="Valeurs réelles"))
        fig.update_layout(legend_title_text="Légende", title=f"Consommation mensuelle pour la MRC de {mrc} - Secteur {sector}", yaxis_title="Valeur (kWh)", xaxis_title="Date")
        fig.show()

        # Save predictions
        y["forecast"] = fore
        y.to_csv(f"arima_predictions/{mrc}_{sector}.csv")