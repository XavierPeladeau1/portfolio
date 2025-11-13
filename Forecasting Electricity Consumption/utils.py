"""
Data Loading and Preprocessing Utilities for Electricity Consumption Forecasting

This module provides functions to load, clean, and prepare electricity consumption data
for forecasting models. It handles:
- Loading raw consumption data from Excel files
- Fetching and processing geolocation data for MRCs
- Retrieving temperature data from Meteostat
- Handling missing values through interpolation and proximity-based filling
- Preparing the final dataset with all necessary features

The main entry point is load_and_prepare_dataset(), which orchestrates all preprocessing steps.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def load_data():
    """
    Load raw electricity consumption data from Excel file.

    Reads the historical consumption data, cleans column names, and converts
    the date column to datetime format.

    Returns:
        DataFrame with columns: mrc, sector, total_kwh, ANNEE_MOIS (datetime)
    """
    df = pd.read_excel("data/consommation-historique-mrc-11mars2024.xlsx")
    df = df.dropna(subset=["MRC_TXT"])
    df = df.set_index(df["ANNEE_MOIS"].infer_objects()).sort_index()
    df = df.rename(columns={"Total (kWh)": "total_kwh", "MRC_TXT": "mrc", "SECTEUR": "sector"})

    df = df.reset_index(drop=True)
    # Convert year-month to proper datetime (add day=01)
    df["ANNEE_MOIS"] = pd.to_datetime(df["ANNEE_MOIS"] + "-01", format="%Y-%m-%d")

    return df


def remove_incomplete_mrc_sectors(df):
    """
    Remove MRC-sector combinations with incomplete or unreliable data.

    Filters out specific MRC-sector pairs that have insufficient historical data
    or data quality issues that would negatively impact model training.

    Args:
        df: DataFrame with 'mrc' and 'sector' columns

    Returns:
        Filtered DataFrame with incomplete MRC-sector combinations removed
    """
    unwanted_mrc_sectors = [
        ("Administration régionale Kativik", "AGRICOLE"),
        ("Administration régionale Kativik", "INDUSTRIEL"),
        ("Caniapiscau", "AGRICOLE"),
        ("Le Golfe-du-Saint-Laurent", "AGRICOLE")
    ]

    for mrc, sector in unwanted_mrc_sectors:
        df = df[~((df["mrc"] == mrc) & (df["sector"] == sector))]

    return df


def interpolate_missing_values(df):
    """
    Interpolate missing consumption values for each MRC-sector time series.

    Uses index-based interpolation to fill gaps in the consumption data,
    processing each MRC-sector combination independently to maintain
    data integrity across different series.

    Args:
        df: DataFrame with 'sector', 'mrc', and 'total_kwh' columns

    Returns:
        DataFrame with interpolated consumption values and new 'sector_mrc' column
    """
    new_df = pd.DataFrame()

    # Create composite key for grouping
    df["sector_mrc"] = df["sector"] + "_" + df["mrc"]

    # Interpolate missing values for each series independently
    for sector_mrc in df["sector_mrc"].unique():
        sector_mrc_df = df[df["sector_mrc"] == sector_mrc].sort_index()
        sector_mrc_df["total_kwh"] = sector_mrc_df["total_kwh"].interpolate(method="index")
        new_df = pd.concat([new_df, sector_mrc_df])

    return new_df


def join_dataframes(df, temp):
    """
    Merge consumption data with temperature data.

    Joins the consumption DataFrame with temperature DataFrame on MRC and date,
    adding temperature features as covariates for forecasting.

    Args:
        df: Consumption DataFrame with 'mrc' and 'ANNEE_MOIS' columns
        temp: Temperature DataFrame with 'mrc' and 'index' (date) columns

    Returns:
        Merged DataFrame with both consumption and temperature data
    """
    temp["index"] = pd.to_datetime(temp["index"], format="%Y-%m-%d")

    joined_df = df.merge(temp, how="left", left_on=["mrc", "ANNEE_MOIS"], right_on=["mrc", "index"])

    return joined_df


def fetch_temperature_data(mapping):
    """
    Fetch monthly temperature data from Meteostat for each MRC.

    Uses the Meteostat library to retrieve historical temperature data based on
    the geographic coordinates of each MRC. Results are cached to avoid repeated API calls.

    Args:
        mapping: DataFrame with MRC names as index and 'lat', 'lng' columns

    Returns:
        DataFrame with columns: mrc, tavg (average temperature), and datetime index
        Data spans from 2016-01-01 to 2023-12-31 at monthly frequency
    """
    from meteostat import Monthly, Point

    # Use cached data if available
    if Path("data/temperature.csv").exists():
        return pd.read_csv("data/temperature.csv", index_col=0)

    df = mapping.reset_index()
    df = df.rename(columns={"index": "mrc"})

    result = pd.DataFrame()
    for i, row in df.iterrows():
        if row["lat"] is None or row["lng"] is None:
            print(f"Skipping {row['mrc']} - missing coordinates")
            continue

        # Fetch temperature data from Meteostat
        point = Point(row["lat"], row["lng"])
        data = Monthly(point, start=datetime.strptime("2016-01-01", "%Y-%m-%d"), end=datetime.strptime("2023-12-31", "%Y-%m-%d"))
        data = data.fetch()
        data["mrc"] = row["mrc"]

        if data.empty:
            print(f"No data found for {row['mrc']}")
            # Create empty template with proper structure
            data = pd.DataFrame(columns=["time", "tavg", "mrc"])
            data["time"] = pd.date_range(start="2016-01-01", end="2023-12-31", freq="MS")
            data.index = data["time"]
            data["tavg"] = None
            data["mrc"] = row["mrc"]
        else:
            # Ensure complete time range
            data = data.reindex(pd.date_range(start="2016-01-01", end="2023-12-31", freq="MS"))
            data["mrc"] = row["mrc"]

        result = pd.concat([result, data])
        print(f"Done with {row['mrc']}")

    # Cache results
    result.to_csv("data/temperature.csv")

    return result

def fill_temperature_data(proximity, temp):
    """
    Fill missing temperature values using proximity-based imputation.

    For each missing temperature value, attempts to fill it using data from the
    closest geographic MRCs. Falls back to global average if no nearby data exists.

    Args:
        proximity: DataFrame with MRC proximity rankings (closest neighbors by distance)
        temp: Temperature DataFrame with potential missing values

    Returns:
        DataFrame with all missing temperature values filled, cached to disk
    """
    # Use cached data if available
    if Path("data/filled_temp.csv").exists():
        return pd.read_csv("data/filled_temp.csv", index_col=0)

    no_na = temp.dropna(subset=["tavg"])
    filled_result = temp.copy()

    # Fill missing values using proximity-based imputation
    for i, row in temp.iterrows():
        if pd.isna(row["tavg"]):
            # Get the three closest MRCs
            closest_mrc = proximity.loc[0, row['mrc']]
            second_closest_mrc = proximity.loc[1, row['mrc']]
            third_closest_mrc = proximity.loc[2, row['mrc']]

            # Try to get temperature from closest MRCs
            closest_mrc_temp = no_na.loc[(no_na["mrc"] == closest_mrc) & (no_na.index == i), "tavg"].mean()
            second_closest_mrc_temp = no_na.loc[(no_na["mrc"] == second_closest_mrc) & (no_na.index == i), "tavg"].mean()
            third_closest_mrc_temp = no_na.loc[(no_na["mrc"] == third_closest_mrc) & (no_na.index == i), "tavg"].mean()

            global_temp_avg = no_na.loc[no_na.index == i, "tavg"].mean()

            # Use cascading fallback: closest -> second closest -> third closest -> global average
            filled_result.loc[i, "tavg"] = closest_mrc_temp
            if pd.isna(closest_mrc_temp):
                filled_result.loc[i, "tavg"] = second_closest_mrc_temp
                if pd.isna(second_closest_mrc_temp):
                    filled_result.loc[i, "tavg"] = third_closest_mrc_temp
                    if pd.isna(third_closest_mrc_temp):
                        print(f"Using global average for {row['mrc']} at index {i}")
                        filled_result.loc[i, "tavg"] = global_temp_avg

    filled_result = filled_result.reset_index()
    # Cache results
    filled_result.to_csv("data/filled_temp.csv")
    filled_result = filled_result.drop(columns=["time"])

    return filled_result

def get_proximity_mapping(mapping):
    """
    Calculate geographic proximity between all MRCs.

    Creates a matrix where each column represents an MRC and contains the indices
    of all other MRCs sorted by distance (closest to farthest).

    Args:
        mapping: DataFrame with MRC names as index and 'lat', 'lng' columns

    Returns:
        DataFrame where each column is an MRC, and values are sorted indices
        of other MRCs by ascending distance (first row = closest MRC)
    """
    import geopandas as gpd

    # Convert to GeoDataFrame with proper projection
    gdf = gpd.GeoDataFrame(mapping, geometry=gpd.points_from_xy(mapping.lng, mapping.lat)).set_crs("ESRI:102003")

    result = pd.DataFrame()
    for i, mrc in gdf.iterrows():
        # Calculate distances from current MRC to all others
        dist_from_mrc = gdf.distance(gdf.loc[i, "geometry"]).sort_values()
        # Store sorted indices (exclude self at position 0)
        result[i] = dist_from_mrc.index[1:]

    return result


def fetch_geolocation_data(df):
    """
    Fetch geographic coordinates (latitude, longitude) for each MRC.

    Uses Google Maps Geocoding API to retrieve coordinates for each unique MRC
    in the dataset. Results are cached to avoid repeated API calls.

    Args:
        df: DataFrame containing 'mrc' column with MRC names

    Returns:
        DataFrame indexed by MRC name with 'lat' and 'lng' columns

    Raises:
        ValueError: If GOOGLE_MAPS_API_KEY is not set in environment variables

    Note:
        Requires GOOGLE_MAPS_API_KEY in .env file.
        Get an API key at https://console.cloud.google.com/google/maps-apis
    """
    import googlemaps

    # Use cached data if available
    if Path("data/geolocation.csv").exists():
        result = pd.read_csv("data/geolocation.csv", index_col=0)
        result.index.name = "mrc"
        return result

    # Check for API key
    api_key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if not api_key:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY not found in environment variables. "
            "Get your API key at https://console.cloud.google.com/google/maps-apis "
            "and set it in the .env file"
        )

    # Fetch coordinates for each MRC
    gmaps = googlemaps.Client(key=api_key)
    mrcs = df["mrc"].unique()
    result = []
    for mrc in mrcs:
        geocode_result = gmaps.geocode(mrc + ", Québec")
        print(geocode_result)
        result.append(geocode_result)

    # Extract coordinates from geocoding results
    mapping = {}
    for i, res in enumerate(result):
        found = False
        for comp in res[0]["address_components"]:
            if "administrative_area_level_3" in comp["types"]:
                found = True
                mapping[mrcs[i]] = res[0]["geometry"]["location"]
                break
        if not found:
            print(f"No administrative_area_level_3 match for {mrcs[i]}")
            mapping[mrcs[i]] = None

    result = pd.DataFrame(mapping).T
    result.index.name = "mrc"

    # Cache results
    result.to_csv("data/geolocation.csv")

    return result


def load_and_prepare_dataset(omit_last_year=False):
    """
    Load and prepare the complete dataset for forecasting models.

    This is the main preprocessing pipeline that orchestrates all data loading,
    cleaning, and feature engineering steps. The workflow is:
    1. Load raw consumption data
    2. Fetch MRC geolocation coordinates
    3. Retrieve temperature data from Meteostat
    4. Fill missing temperature values using proximity-based imputation
    5. Merge consumption and temperature data
    6. Remove incomplete MRC-sector combinations
    7. Interpolate missing consumption values
    8. Create derived features (log transform, sector_mrc composite key)
    9. Optionally filter to exclude test year

    Args:
        omit_last_year: If True, excludes 2023 data (useful for hyperparameter tuning)

    Returns:
        DataFrame with columns:
        - mrc: MRC name
        - sector: Economic sector
        - date: Monthly datetime index
        - total_kwh: Raw consumption values
        - log_volume: Log-transformed consumption (for modeling)
        - tavg: Average monthly temperature
        - sector_mrc: Composite key (sector_mrc format)
    """
    # Step 1: Load raw data
    df = load_data()

    # Step 2: Fetch geolocation data for each MRC
    mapping = fetch_geolocation_data(df)

    # Manual correction for Les Appalaches (missing from geocoding results)
    mapping.loc["Les Appalaches", "lat"] = 46.374163
    mapping.loc["Les Appalaches", "lng"] = -70.440328

    # Step 3-4: Fetch and fill temperature data
    temp = fetch_temperature_data(mapping)
    proximity = get_proximity_mapping(mapping)
    temp = fill_temperature_data(proximity, temp)

    # Step 5-7: Merge data and handle missing values
    df = join_dataframes(df, temp)
    df = remove_incomplete_mrc_sectors(df)
    df = interpolate_missing_values(df)

    # Step 8: Clean up columns and create features
    df = df.drop(columns=["REGION_ADM_QC_TXT", "index", "tmin", "tmax", "prcp", "wspd", "pres", "tsun"])
    df = df.rename(columns={"ANNEE_MOIS": "date"})

    # Create log-transformed consumption (helps with model stability)
    df["log_volume"] = np.log(df.total_kwh)
    df = df.sort_values(['sector_mrc', 'date'])

    # Convert to float32 for memory efficiency
    df[["total_kwh", "log_volume", "tavg"]] = df[["total_kwh", "log_volume", "tavg"]].astype(np.float32)

    # Step 9: Optionally filter out test year
    if omit_last_year:
        df = df[df["date"] < pd.to_datetime("2023-01-01")]

    return df