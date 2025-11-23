"""
Diurnal Profile Analysis Module for Greenhouse Optimization Project.

This module computes 24-hour diurnal profiles for each greenhouse and analyzes
environmental variables (temperature, humidity, CO₂) against comfort bands.
Produces statistical summaries and visualization plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import os
from pathlib import Path

from load_data import load_greenhouse_data


# Comfort bands for environmental variables
COMFORT_BANDS = {
    'temperature': {'min': 18.0, 'max': 28.0},  # Celsius
    'humidity': {'min': 60.0, 'max': 90.0},     # Percentage
    'co2': {'min': 400.0, 'max': 800.0}         # ppm
}

# Greenhouse identifiers
GREENHOUSES = ['A', 'B', 'C', 'D']

# Variables to analyze
ENVIRONMENTAL_VARIABLES = ['temperature', 'humidity', 'co2']


def extract_hour_from_timestamp(df: pd.DataFrame, 
                                timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    Extract hour of day from timestamp column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with timestamp column
    timestamp_col : str
        Name of timestamp column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added 'hour' column (0-23)
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Extract hour
    df['hour'] = df[timestamp_col].dt.hour
    
    return df


def compute_diurnal_profiles(df: pd.DataFrame,
                            greenhouse_col: str = 'greenhouse',
                            variables: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Compute 24-hour diurnal profiles (mean ± std) for each greenhouse.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with timestamp, greenhouse, and environmental variables
    greenhouse_col : str
        Name of greenhouse identifier column
    variables : list of str, optional
        List of environmental variables to analyze. If None, uses default list.
        
    Returns:
    --------
    dict
        Dictionary with keys as variable names, values as DataFrames with
        columns: greenhouse, hour, mean, std, count
    """
    if variables is None:
        variables = ENVIRONMENTAL_VARIABLES
    
    # Ensure hour column exists
    if 'hour' not in df.columns:
        df = extract_hour_from_timestamp(df)
    
    profiles = {}
    
    for var in variables:
        if var not in df.columns:
            print(f"Warning: Variable '{var}' not found in dataframe. Skipping.")
            continue
        
        # Group by greenhouse and hour, compute statistics
        grouped = df.groupby([greenhouse_col, 'hour'])[var].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).reset_index()
        
        # Fill NaN std values with 0 (when only one observation)
        grouped['std'] = grouped['std'].fillna(0.0)
        
        profiles[var] = grouped
    
    return profiles


def compute_comfort_band_statistics(df: pd.DataFrame,
                                   greenhouse_col: str = 'greenhouse',
                                   variables: List[str] = None,
                                   comfort_bands: Dict = None) -> pd.DataFrame:
    """
    Compute percentage of time inside, below, and above comfort bands.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with greenhouse and environmental variables
    greenhouse_col : str
        Name of greenhouse identifier column
    variables : list of str, optional
        List of environmental variables to analyze
    comfort_bands : dict, optional
        Dictionary with comfort band definitions. If None, uses default.
        
    Returns:
    --------
    pd.DataFrame
        Summary dataframe with columns:
        - greenhouse
        - variable
        - pct_inside_band
        - pct_below_band
        - pct_above_band
    """
    if variables is None:
        variables = ENVIRONMENTAL_VARIABLES
    
    if comfort_bands is None:
        comfort_bands = COMFORT_BANDS
    
    results = []
    
    for var in variables:
        if var not in df.columns:
            continue
        
        if var not in comfort_bands:
            print(f"Warning: No comfort band defined for '{var}'. Skipping.")
            continue
        
        band_min = comfort_bands[var]['min']
        band_max = comfort_bands[var]['max']
        
        # Group by greenhouse
        for greenhouse in df[greenhouse_col].unique():
            gh_data = df[df[greenhouse_col] == greenhouse][var].dropna()
            
            if len(gh_data) == 0:
                continue
            
            # Count observations
            total = len(gh_data)
            inside = ((gh_data >= band_min) & (gh_data <= band_max)).sum()
            below = (gh_data < band_min).sum()
            above = (gh_data > band_max).sum()
            
            # Calculate percentages
            pct_inside = (inside / total) * 100.0
            pct_below = (below / total) * 100.0
            pct_above = (above / total) * 100.0
            
            results.append({
                'greenhouse': greenhouse,
                'variable': var,
                'pct_inside_band': pct_inside,
                'pct_below_band': pct_below,
                'pct_above_band': pct_above,
                'total_observations': total
            })
    
    summary_df = pd.DataFrame(results)
    return summary_df


def plot_diurnal_profile(profile_df: pd.DataFrame,
                        variable: str,
                        greenhouse: str,
                        comfort_band: Dict = None,
                        save_path: str = None,
                        figsize: Tuple[int, int] = (10, 6)):
    """
    Plot 24-hour diurnal profile with mean ± std and comfort band.
    
    Parameters:
    -----------
    profile_df : pd.DataFrame
        Profile dataframe with columns: hour, mean, std
    variable : str
        Name of environmental variable
    greenhouse : str
        Greenhouse identifier
    comfort_band : dict, optional
        Comfort band definition {'min': float, 'max': float}
    save_path : str, optional
        Path to save figure. If None, displays plot.
    figsize : tuple
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    hours = profile_df['hour'].values
    means = profile_df['mean'].values
    stds = profile_df['std'].values
    
    # Plot mean line
    ax.plot(hours, means, 'o-', linewidth=2, markersize=6, 
           label='Mean', color='#2ecc71')
    
    # Plot mean ± std shaded area
    ax.fill_between(hours, means - stds, means + stds, 
                    alpha=0.3, color='#3498db', label='Mean ± 1 SD')
    
    # Plot comfort band if provided
    if comfort_band is not None:
        band_min = comfort_band['min']
        band_max = comfort_band['max']
        ax.axhspan(band_min, band_max, alpha=0.2, color='green', 
                  label=f"Comfort Band ({band_min}-{band_max})")
        ax.axhline(band_min, color='green', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(band_max, color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{variable.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    ax.set_title(f'24-Hour Diurnal Profile: {variable.title()} - Greenhouse {greenhouse}',
                fontsize=14, fontweight='bold')
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
        plt.close()
    else:
        plt.show()


def save_all_diurnal_plots(profiles: Dict[str, pd.DataFrame],
                          greenhouse_col: str = 'greenhouse',
                          comfort_bands: Dict = None,
                          figures_dir: str = 'figures'):
    """
    Save all diurnal profile plots for each greenhouse and variable.
    
    Parameters:
    -----------
    profiles : dict
        Dictionary of profile dataframes from compute_diurnal_profiles()
    greenhouse_col : str
        Name of greenhouse column in profile dataframes
    comfort_bands : dict, optional
        Comfort band definitions
    figures_dir : str
        Directory to save figures
    """
    if comfort_bands is None:
        comfort_bands = COMFORT_BANDS
    
    for var, profile_df in profiles.items():
        for greenhouse in profile_df[greenhouse_col].unique():
            gh_profile = profile_df[profile_df[greenhouse_col] == greenhouse].copy()
            gh_profile = gh_profile.sort_values('hour')
            
            # Get comfort band for this variable
            band = comfort_bands.get(var, None)
            
            # Generate filename
            filename = f'daily_profile_{var}_greenhouse_{greenhouse}.png'
            save_path = os.path.join(figures_dir, filename)
            
            # Plot and save
            plot_diurnal_profile(gh_profile, var, greenhouse, band, save_path)


def print_summary_table(summary_df: pd.DataFrame):
    """
    Print formatted summary table to stdout.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary dataframe from compute_comfort_band_statistics()
    """
    print("\n" + "=" * 80)
    print("COMFORT BAND STATISTICS SUMMARY")
    print("=" * 80)
    
    # Group by variable for better readability
    for var in summary_df['variable'].unique():
        var_data = summary_df[summary_df['variable'] == var].copy()
        
        print(f"\n{var.upper().replace('_', ' ')}")
        print("-" * 80)
        print(f"{'Greenhouse':<12} {'Inside Band':<15} {'Below Band':<15} {'Above Band':<15} {'Total Obs':<12}")
        print("-" * 80)
        
        for _, row in var_data.iterrows():
            print(f"{row['greenhouse']:<12} "
                 f"{row['pct_inside_band']:>6.2f}%      "
                 f"{row['pct_below_band']:>6.2f}%      "
                 f"{row['pct_above_band']:>6.2f}%      "
                 f"{int(row['total_observations']):<12}")
    
    print("\n" + "=" * 80)


def main():
    """
    Main function to run complete diurnal profile analysis.
    
    Loads data, computes profiles and statistics, prints summary,
    and saves all plots.
    """
    print("=" * 80)
    print("DIURNAL PROFILE ANALYSIS - GREENHOUSE ENVIRONMENTAL VARIABLES")
    print("=" * 80)
    
    # 1. Load data
    print("\n1. Loading greenhouse data...")
    
    # Try multiple possible paths
    possible_paths = [
        'data/greenhouse_data.xlsx',
        '../data/greenhouse_data.xlsx',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'greenhouse_data.xlsx')
    ]
    
    df = None
    for filepath in possible_paths:
        try:
            df = load_greenhouse_data(filepath)
            print(f"   Loaded {len(df)} rows from {filepath}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError(
            "Could not find greenhouse_data.xlsx. Tried paths:\n" + 
            "\n".join(f"  - {p}" for p in possible_paths)
        )
    
    # Check required columns
    required_cols = ['timestamp', 'greenhouse', 'temperature', 'humidity', 'co2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 2. Extract hour from timestamp
    print("\n2. Extracting hour of day from timestamps...")
    df = extract_hour_from_timestamp(df)
    print(f"   Hour range: {df['hour'].min()} - {df['hour'].max()}")
    
    # 3. Compute diurnal profiles
    print("\n3. Computing 24-hour diurnal profiles...")
    profiles = compute_diurnal_profiles(df)
    print(f"   Computed profiles for {len(profiles)} variables")
    
    # 4. Compute comfort band statistics
    print("\n4. Computing comfort band statistics...")
    summary_df = compute_comfort_band_statistics(df)
    print(f"   Computed statistics for {len(summary_df)} greenhouse-variable pairs")
    
    # 5. Print summary table
    print_summary_table(summary_df)
    
    # 6. Generate and save plots
    print("\n5. Generating diurnal profile plots...")
    # Determine figures directory path (handle both direct run and module run)
    if os.path.exists('figures'):
        figures_dir = 'figures'
    elif os.path.exists('../figures'):
        figures_dir = '../figures'
    else:
        # Create figures directory in project root
        project_root = os.path.dirname(os.path.dirname(__file__)) if '__file__' in globals() else '.'
        figures_dir = os.path.join(project_root, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
    
    save_all_diurnal_plots(profiles, figures_dir=figures_dir)
    print(f"   Saved plots to {figures_dir}/ directory")
    
    # 7. Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  - Analyzed {len(profiles)} environmental variables")
    print(f"  - Processed {len(df['greenhouse'].unique())} greenhouses")
    print(f"  - Generated {len(profiles) * len(df['greenhouse'].unique())} plots")
    print(f"  - All plots saved to: figures/")
    print()


if __name__ == '__main__':
    main()

