"""
Data loading and preprocessing module for greenhouse optimization project.

This module handles loading the greenhouse_data.xlsx dataset, cleaning
physical outliers, and forward-filling missing values per greenhouse group.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


# Physical limits for sensor validation
PHYS_LIMITS = {
    'temperature': {'min': 5.0, 'max': 50.0},  # Celsius
    'humidity': {'min': 0.0, 'max': 100.0},   # Percentage
    'co2': {'min': 200.0, 'max': 2000.0},     # ppm
    'pH': {'min': 4.0, 'max': 9.0},           # pH units
    'EC': {'min': 0.0, 'max': 5.0}            # mS/cm
}


def load_greenhouse_data(filepath: str = 'data/greenhouse_data.xlsx') -> pd.DataFrame:
    """
    Load greenhouse data from Excel file.
    
    Parameters:
    -----------
    filepath : str
        Path to the greenhouse_data.xlsx file
        
    Returns:
    --------
    pd.DataFrame
        Raw greenhouse data
    """
    try:
        df = pd.read_excel(filepath)
        print(f"Loaded {len(df)} rows from {filepath}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure greenhouse_data.xlsx exists.")


def clean_physical_outliers(df: pd.DataFrame, 
                           limits: Dict = None) -> pd.DataFrame:
    """
    Remove physical outliers based on sensor limits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with sensor columns
    limits : dict, optional
        Physical limits dictionary. If None, uses PHYS_LIMITS.
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers set to NaN
    """
    if limits is None:
        limits = PHYS_LIMITS
    
    df_clean = df.copy()
    initial_count = len(df_clean)
    
    # Clean each sensor column
    for sensor, bounds in limits.items():
        if sensor in df_clean.columns:
            # Set outliers to NaN
            mask = (df_clean[sensor] < bounds['min']) | (df_clean[sensor] > bounds['max'])
            n_outliers = mask.sum()
            if n_outliers > 0:
                print(f"Removed {n_outliers} outliers from {sensor}")
                df_clean.loc[mask, sensor] = np.nan
    
    return df_clean


def forward_fill_missing(df: pd.DataFrame, 
                        group_col: str = 'greenhouse_id') -> pd.DataFrame:
    """
    Forward-fill missing values per greenhouse group.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    group_col : str
        Column name to group by (e.g., 'greenhouse_id')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with forward-filled missing values
    """
    df_filled = df.copy()
    
    if group_col in df_filled.columns:
        # Group by greenhouse and forward-fill within each group
        df_filled = df_filled.groupby(group_col).ffill()
        print(f"Forward-filled missing values per {group_col}")
    else:
        # If no group column, forward-fill globally
        df_filled = df_filled.ffill()
        print("Forward-filled missing values globally")
    
    return df_filled


def preprocess_greenhouse_data(filepath: str = 'data/greenhouse_data.xlsx',
                               group_col: str = 'greenhouse_id') -> Tuple[pd.DataFrame, Dict]:
    """
    Complete preprocessing pipeline: load, clean outliers, forward-fill.
    
    Parameters:
    -----------
    filepath : str
        Path to the greenhouse_data.xlsx file
    group_col : str
        Column name to group by for forward-filling
        
    Returns:
    --------
    Tuple[pd.DataFrame, Dict]
        Clean dataframe and physical limits dictionary
    """
    # Load data
    df = load_greenhouse_data(filepath)
    
    # Clean physical outliers
    df_clean = clean_physical_outliers(df)
    
    # Forward-fill missing values
    df_filled = forward_fill_missing(df_clean, group_col=group_col)
    
    print(f"\nPreprocessing complete:")
    print(f"  Initial rows: {len(df)}")
    print(f"  Final rows: {len(df_filled)}")
    print(f"  Missing values remaining: {df_filled.isna().sum().sum()}")
    
    return df_filled, PHYS_LIMITS


if __name__ == '__main__':
    # Example usage
    df, limits = preprocess_greenhouse_data()
    print("\nFirst few rows:")
    print(df.head())
    print("\nData info:")
    print(df.info())

