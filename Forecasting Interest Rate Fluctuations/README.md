# Interest Rate Forecasting with Time Series Models

Multi-horizon forecasting of US 10-year Treasury yields using deep learning and statistical models. Compares different modeling approaches with proper chronological validation and baseline benchmarking.
Initial work done as part of the Défi IA 2025 competition organized by Caisse de Dépôt et Placement du Québec (CDPQ), which my two colleages and I won in January 2025.

## Key Features

- **Multi-step forecasting**: 1 to 36 months ahead predictions
- **Probabilistic outputs**: Quantile forecasts with uncertainty intervals
- **Rigorous evaluation**: Walk-forward validation, chronological splits, baseline comparisons
- **Public data**: FRED API integration for reproducibility

## Tech Stack

- **Models**: PyTorch (LSTM, Transformers)
- **Framework**: Darts time series library
- **Data**: Federal Reserve Economic Data (FRED) API

## Setup

1. Install dependencies:
   ```bash
   pip install fredapi python-dotenv pandas numpy darts pytorch-lightning
   ```

2. Get a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html)

3. Create `.env` file:
   ```bash
   FRED_API_KEY=your_api_key_here
   ```

4. Run the EDA notebook:
   ```bash
   jupyter notebook 01_eda.ipynb
   ```

## Project Structure

```
├── utils.py                      # Core utilities (data loading, FRED API, evaluation)
├── 01_eda.ipynb                  # Exploratory data analysis & stationarity tests
├── 02_naive.ipynb                # Baseline models (random walk, moving average)
├── 03_AR.ipynb                   # Autoregressive models
├── 04_variable_selection.ipynb   # Feature engineering & selection
├── 05_Lasso.ipynb                # Lasso regression for sparse models
├── 06_XGB.ipynb                  # XGBoost with lag features
├── 07_VAR.ipynb                  # Vector Autoregression (multivariate)
├── 08_tft.ipynb                  # Temporal Fusion Transformer
├── 09_nhits.ipynb                # N-HiTS (neural hierarchical interpolation)
├── 10_hp_optim.py                # Hyperparameter optimization script
├── plot_presentation.ipynb       # Visualization & results summary
└── data/
    └── sf_fed/                   # San Francisco Fed news sentiment data
```

Notebooks follow a logical progression from baseline → classical ML → deep learning.

## Results

Performance comparison across forecast horizons (lower is better):

| Model | Category | 1 Month |  | 3 Months |  | 12 Months |  | 36 Months |  |
|-------|----------|---------|---|----------|---|-----------|---|-----------|---|
|       |          | MAPE | MAE | MAPE | MAE | MAPE | MAE | MAPE | MAE |
| Naive | Extrapolation | 7.7% | 0.146 | 18.2% | 0.321 | 53.6% | 0.909 | 84.4% | 1.358 |
| AR(12) | Extrapolation | 7.6% | 0.147 | 16.9% | 0.304 | 52.7% | 0.987 | 65.0% | 1.420 |
| VAR | Multivariate Regression | 17.3% | 0.236 | 38.7% | 0.596 | 90.5% | 1.616 | 57.2% | 1.668 |
| Lasso | Univariate Regression | 7.4% | 0.146 | 16.2% | 0.308 | 43.2% | 0.859 | 53.5% | 1.273 |
| N-HiTS | Non-parametric | **7.2%** | **0.143** | 16.4% | 0.319 | 44.8% | 0.926 | 54.9% | 1.444 |
| TFT | Non-parametric | 7.3% | 0.144 | **16.1%** | **0.304** | **42.6%** | **0.791** | **53.0%** | **1.154** |

**Key findings:**
- **Best short-term (1 month)**: N-HiTS achieves 7.2% MAPE, outperforming all models including naive baseline
- **Best long-term (36 months)**: TFT excels at extended horizons with 53.0% MAPE vs. 84.4% for naive
- **Classical ML**: Lasso regression competitive with deep learning at short/medium horizons, offering better interpretability

## Data Sources

- **Primary**: FRED API (Treasury yields, unemployment, CPI, Fed funds rate, S&P 500, GDP, employment, inflation expectations)
- **Sentiment**: SF Fed Daily News Sentiment Index (public dataset)

All data sources are publicly available for reproducibility.
