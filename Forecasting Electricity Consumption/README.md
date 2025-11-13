# Forecasting Electricity Consumption

Time series forecasting of electricity consumption patterns across Quebec regions using classical statistical models and deep learning approaches. Developed as part of MATH60630 (Machine Learning II) coursework at HEC Montréal.

## Overview

This project analyzes and forecasts electricity consumption for different Municipal Regional Counties (MRCs) and sectors in Quebec using historical data from Hydro-Québec. The analysis incorporates weather data (temperature) and geographic information to improve forecast accuracy across various consumer segments (residential, commercial, industrial, agricultural).

## Tech Stack

- **Statistical Models**: ARIMA/SARIMAX
- **Deep Learning**: Temporal Convolutional Networks (TCN)
- **Data Processing**: Python, Pandas, NumPy
- **Geospatial**: Google Maps API, Geopandas
- **Weather Data**: Meteostat API
- **Visualization**: Matplotlib, Jupyter Notebooks

## Features

### Data Processing
- Historical electricity consumption data aggregation by MRC and sector
- Temperature data integration using geographic coordinates
- Missing value interpolation based on spatial proximity using Google Maps API
- Feature engineering with seasonal and weather variables

### Modeling Approaches

**Classical Statistical Models (SARIMAX)**
- Captures seasonal patterns and trends
- Incorporates exogenous variables (temperature, sector type)
- Provides interpretable coefficients

**Temporal Convolutional Networks (TCN)**
- Parallel processing of time series
- Long-range dependency modeling
- Efficient computation for long sequences

## Project Structure

```
├── arima.py                                    # SARIMAX model implementation
├── tcn.py                                      # Temporal Convolutional Network model
├── utils.py                                    # Data loading and preprocessing utilities
├── make_figures.ipynb                          # Visualization and results analysis
├── consommation-historique-mrc-11mars2024.xlsx # Raw Hydro-Québec consumption data
├── dataset.csv                                 # Processed dataset
├── data/
│   ├── geolocation.csv                         # MRC geographic coordinates
│   ├── temperature.csv                         # Historical temperature data
│   └── filled_temp.csv                         # Interpolated temperature data
└── .env.example                                # Environment variables template
```

## Setup

### Install Required Dependencies

```bash
pip install pandas numpy scipy scikit-learn statsmodels patsy torch darts pytorch-lightning optuna meteostat googlemaps geopandas pyproj matplotlib plotly openpyxl python-dotenv jupyter
```

### Configuration

1. **Set up API access:**
   - Copy `.env.example` to `.env`
   - Add your Google Maps API key (required for geolocation fetching if running from scratch)
   ```bash
   GOOGLE_MAPS_API_KEY=your_api_key_here
   ```
   Note: Pre-computed geolocation and temperature data are included in the `data/` directory, so the API key is only needed if regenerating from scratch.

2. **Run the analysis:**
   - `python arima.py` - Train SARIMAX models
   - `python tcn.py` - Train TCN models
   - `jupyter notebook make_figures.ipynb` - Generate visualizations

## Data Sources

- **Hydro-Québec**: Historical electricity consumption by MRC and sector
- **Meteostat**: Historical temperature data for Quebec regions
- **Google Maps API**: Geographic coordinates for MRC spatial analysis

## Key Insights

The project compares forecasting performance across different model architectures:
- **SARIMAX** excels at capturing clear seasonal patterns with interpretable parameters
- **TCN** demonstrates strong performance in modeling complex temporal dependencies

Model selection depends on the specific MRC-sector combination, data availability, and forecast horizon requirements.

## Author

Xavier Péladeau