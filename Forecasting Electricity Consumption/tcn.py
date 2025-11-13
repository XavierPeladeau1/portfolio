"""
Temporal Convolutional Network (TCN) for Electricity Consumption Forecasting

This script trains a TCN model to forecast monthly electricity consumption for different
MRCs (regional municipalities) and economic sectors in Quebec. TCNs are deep learning
models that use dilated causal convolutions to capture long-term temporal dependencies.

The workflow consists of:
1. Loading and preparing time series data for all MRC-sector combinations
2. Training a single TCN model on all series simultaneously (multi-series learning)
3. Evaluating model performance across all series
4. Generating predictions for filtered MRCs and saving results

Note: The script includes a commented hyperparameter optimization section using Optuna.
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, smape, rmse
from darts.models import TCNModel
from darts.utils.likelihood_models import LaplaceLikelihood
from optuna.integration import PyTorchLightningPruningCallback
from optuna.visualization import (plot_contour, plot_optimization_history,
                                  plot_param_importances)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
from utils import *

# Disable pandas scientific notation for cleaner output
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Define MRC filter for predictions - modify this list to change which MRCs to generate predictions for
MRC_FILTER = ["Drummond", "Les Etchemins"]

# ============================================================================
# Dataset Preparation
# ============================================================================
# Load all data (no filter) for training - the model learns from all MRC-sector combinations
df = load_and_prepare_dataset(omit_last_year=False)

# Create TimeSeries objects for Darts library
# - series: log-transformed consumption values
# - temp_series: temperature data (used as past covariates)
series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='log_volume', time_col='date')
temp_series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='tavg', time_col='date')

# Note: Manual scaling is disabled - using built-in Scaler in model's add_encoders instead
# scaler = Scaler()
# temp_scaler = Scaler()
# series = scaler.fit_transform(series)
# temp_series = temp_scaler.fit_transform(temp_series)

def build_fit_tcn_model(
    in_len,
    out_len,
    kernel_size,
    num_filters,
    weight_norm,
    dilation_base,
    dropout,
    lr,
    num_layers,
    likelihood=None,
    callbacks=None,
):
    """
    Build, train, and return a TCN model with specified hyperparameters.

    Args:
        in_len: Input sequence length (number of past time steps)
        out_len: Output sequence length (forecast horizon)
        kernel_size: Size of convolutional kernel
        num_filters: Number of filters in each convolutional layer
        weight_norm: Whether to use weight normalization
        dilation_base: Base for exponential dilation in TCN layers
        dropout: Dropout rate for regularization
        lr: Learning rate for optimizer
        num_layers: Number of TCN layers
        likelihood: Optional likelihood model for probabilistic forecasting
        callbacks: Optional additional callbacks for training

    Returns:
        Trained TCNModel instance (reloaded from best checkpoint)
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Fixed training parameters
    BATCH_SIZE = 32
    MAX_N_EPOCHS = 7
    NR_EPOCHS_VAL_PERIOD = 1

    # Setup early stopping to prevent overfitting
    early_stopper = EarlyStopping("val_loss", min_delta=0.0001, patience=3)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks = [early_stopper] + callbacks

    # Configure GPU/CPU training
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "gpus": -1,
            "auto_select_gpus": True,
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0

    # Build the TCN model
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_N_EPOCHS,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
        kernel_size=kernel_size,
        num_filters=num_filters,
        num_layers=num_layers,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        likelihood=likelihood,
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="tcn_model",
        force_reset=True,
        save_checkpoints=True,
        add_encoders={"transformer": Scaler()}  # Built-in scaling
    )

    # Prepare train/validation splits
    # Hold out last 12 months for validation
    train = [s[: -12] for s in series]
    temp_train = [s[: -12] for s in temp_series]
    print(f"Training end date: {train[0].end_time()}")

    # Validation set includes extra months to provide context for the model
    model_val_set = [s[-18:] for s in series]
    model_val_temp = [s[-18:] for s in temp_series]
    print(f"Validation period: {model_val_set[0].start_time()} to {model_val_set[0].end_time()}")

    # Train the model on all series
    model.fit(
        series=train,
        past_covariates=temp_train,
        val_series=model_val_set,
        val_past_covariates=model_val_temp,
        dataloader_kwargs={"num_workers": num_workers},
    )

    # Reload best model from checkpoint
    model = TCNModel.load_from_checkpoint("tcn_model")

    return model


# ============================================================================
# Model Training
# ============================================================================
# Train TCN with best hyperparameters (found through Optuna optimization - see below)
model = build_fit_tcn_model(
    in_len=12,          # Use 12 months of history
    out_len=1,          # Forecast 1 month ahead
    kernel_size=8,
    num_filters=24,
    weight_norm=True,
    dilation_base=4,
    dropout=0.4,
    lr=2e-4,
    num_layers=3,
)


def evaluate(model, series, temp_series):
    """
    Evaluate model performance on test set (2023) using historical forecasts.

    Generates 1-month-ahead rolling forecasts for each time step in 2023,
    computes MAPE and RMSE metrics, and visualizes predictions for sample series.

    Args:
        model: Trained TCNModel instance
        series: List of TimeSeries (consumption data)
        temp_series: List of TimeSeries (temperature covariates)

    Returns:
        Tuple of (MAPE values, RMSE values) for all series
    """
    # Generate rolling 1-month-ahead forecasts for 2023
    preds = model.historical_forecasts(
        series=series,
        past_covariates=temp_series,
        forecast_horizon=1,
        stride=1,
        retrain=False,  # Use fixed trained model
        start=pd.Timestamp("2023-01-01"),
        verbose=False
    )

    # Reverse log transformation to get actual kWh values
    series = [s.map(np.exp) for s in series]
    preds = [s.map(np.exp) for s in preds]

    # Calculate error metrics
    smapes = mape(series, preds)
    rmses = rmse(series, preds)
    print("MAPE: {:.2f} +- {:.2f}".format(np.mean(smapes), np.std(smapes)))
    print("RMSE: {:.2f} +- {:.2f}".format(np.mean(rmses), np.std(rmses)))

    # Visualize predictions for 20 random series
    for i in np.random.choice(range(len(series)), 20):
        plt.figure(figsize=(10, 6))
        series[i].plot(label="actual")
        preds[i].plot(label="forecast")
        plt.title(f"MAPE: {smapes[i]:.2f} - Sector: {series[i].static_covariates['sector_mrc'].iloc[0]}")
        plt.legend()
        plt.scatter(series[i].time_index, series[i].values(), color='black', s=10)
        plt.ticklabel_format(style='plain', axis='y')
        plt.ylim(0, 1.1 * max(series[i].values()))
        plt.show()

    return smapes, rmses


# ============================================================================
# Model Evaluation
# ============================================================================
# Evaluate on all series and compute metrics
mapes, rmses = evaluate(model, series, temp_series)
sectors = [s.static_covariates["sector_mrc"].iloc[0].split("_")[0] for s in series]

results = pd.DataFrame(dict(sector=sectors, mape=mapes, rmse=rmses))

# Print aggregated results by sector and overall
print(results.groupby("sector").mean())
print(f"Globally, {results[["mape", "rmse"]].mean()}")


def get_forecast(mrc, series, temp_series, model):
    """
    Generate forecasts for all sectors within a specific MRC.

    Args:
        mrc: MRC name to filter for
        series: List of all TimeSeries (consumption data)
        temp_series: List of all TimeSeries (temperature covariates)
        model: Trained TCNModel instance

    Returns:
        List of prediction TimeSeries for the specified MRC (one per sector)
    """
    # Filter series for the specified MRC
    series = [s for s in series if s.static_covariates["sector_mrc"].str.endswith(mrc).all()]
    temp_series = [s for s in temp_series if s.static_covariates["sector_mrc"].str.endswith(mrc).all()]

    # Generate forecasts
    preds = model.historical_forecasts(
        series=series,
        past_covariates=temp_series,
        forecast_horizon=1,
        stride=1,
        retrain=False,
        start=pd.Timestamp("2023-01-01"),
        verbose=False
    )

    # Reverse log transformation
    series = [s.map(np.exp) for s in series]
    preds = [s.map(np.exp) for s in preds]

    return preds


# ============================================================================
# Generate and Save Predictions for Filtered MRCs
# ============================================================================
# Generate predictions only for MRCs specified in MRC_FILTER
results_df = pd.DataFrame()
for mrc in MRC_FILTER:
    mrc_preds = get_forecast(mrc, series, temp_series, model)
    for pred in mrc_preds:
        # Each column is a sector_mrc combination
        results_df[pred.static_covariates_values()[0][0]] = pred.pd_dataframe()

# Save predictions to CSV
results_df.to_csv("tcn_preds.csv")


# ============================================================================
# Hyperparameter Optimization (Optional)
# ============================================================================
# This section uses Optuna to search for optimal TCN hyperparameters.
# The hyperparameters found through this optimization are used in the
# model training section above.
#
# To run hyperparameter search:
# 1. Uncomment the code below
# 2. Adjust the timeout parameter as needed (currently 2 hours)
# 3. Run the script
#
# The search explores: input length, kernel size, number of filters,
# weight normalization, dilation base, dropout, learning rate, and number of layers.

# df = load_and_prepare_dataset(omit_last_year=True)  # Omit 2023 for optimization

# # Prepare dataset for optimization (excluding 2023)
# # series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='log_volume', time_col='date')
# # temp_series = TimeSeries.from_group_dataframe(df, group_cols=['sector_mrc'], value_cols='tavg', time_col='date')

# # scaler = Scaler(global_fit=False)
# # temp_scaler = Scaler()
# # series = scaler.fit_transform(series)
# # temp_series = temp_scaler.fit_transform(temp_series)


# def objective(trial):
#     """
#     Optuna objective function for hyperparameter optimization.
#
#     This function is called by Optuna for each trial. It suggests hyperparameters,
#     trains a model, evaluates it on 2022 validation data, and returns the SMAPE.
#
#     Args:
#         trial: Optuna trial object
#
#     Returns:
#         Mean SMAPE across all series (lower is better)
#     """
#     # Fixed parameters for optimization
#     months_in = 12
#     months_out = 1
#
#     # Hyperparameters to optimize
#     kernel_size = trial.suggest_int("kernel_size", 3, 11)
#     num_filters = trial.suggest_int("num_filters", 6, 25)
#     weight_norm = trial.suggest_categorical("weight_norm", [False, True])
#     dilation_base = trial.suggest_int("dilation_base", 2, 4)
#     dropout = trial.suggest_float("dropout", 0.1, 0.4)
#     lr = trial.suggest_float("lr", 0.0001, 0.005, log=True)
#     num_layers = trial.suggest_int("num_layers", 1, 3)
#
#     # Ensure kernel_size is valid
#     kernel_size = min(kernel_size, months_in) - 1
#
#     # Train model with suggested hyperparameters
#     model = build_fit_tcn_model(
#         in_len=months_in,
#         out_len=months_out,
#         kernel_size=kernel_size,
#         num_filters=num_filters,
#         weight_norm=weight_norm,
#         dilation_base=dilation_base,
#         dropout=dropout,
#         lr=lr,
#         num_layers=num_layers,
#         callbacks=[],
#     )
#
#     # Evaluate on 2022 validation set
#     preds = model.historical_forecasts(
#         series=series,
#         past_covariates=temp_series,
#         forecast_horizon=1,
#         stride=1,
#         retrain=False,
#         start=pd.Timestamp("2022-01-01"),
#         verbose=True,
#     )
#     smapes = smape(series, preds)
#     smape_val = np.mean(smapes)
#
#     return smape_val if smape_val != np.nan else float("inf")

# def print_callback(study, trial):
#     """Callback to print progress during optimization."""
#     print(f"Current value: {trial.value}, Current params: {trial.params}")
#     print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
#
#
# # Create optimization study (minimize SMAPE)
# study = optuna.create_study(direction="minimize")
#
# # Run optimization for 2 hours with progress callback
# study.optimize(objective, timeout=7200, callbacks=[print_callback])
#
# # Print final results
# print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
#
# # Best hyperparameters found (used in model training above):
# # - in_len: 12
# # - out_len: 1
# # - kernel_size: 8
# # - num_filters: 24
# # - weight_norm: True
# # - dilation_base: 4
# # - dropout: 0.4
# # - lr: 2e-4
# # - num_layers: 3

