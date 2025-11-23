"""
Exploratory Data Analysis Script

This script performs exploratory data analysis on the greenhouse dataset, including:
- Data loading and preprocessing
- Correlation analysis
- Scatter plots for key variable relationships
- Figures 1-9 from the article
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import preprocess_greenhouse_data, PHYS_LIMITS
from eda_correlations import run_eda_pipeline, compute_correlations, plot_scatter
from utils_plotting import setup_plot_style, save_figure
from report_excel import save_eda_to_excel

# Setup plotting style
setup_plot_style()
plt.rcParams['figure.figsize'] = (12, 8)


def main():
    """Main function for exploratory data analysis."""
    
    print("=" * 60)
    print("Exploratory Data Analysis - Greenhouse Dataset")
    print("=" * 60)
    
    # 1. Load and Preprocess Data
    print("\n1. Loading and preprocessing data...")
    df, limits = preprocess_greenhouse_data('data/greenhouse_data.xlsx')
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    # 2. Summary Statistics
    print("\n" + "=" * 60)
    print("2. Summary Statistics")
    print("=" * 60)
    print(df.describe())
    
    # 3. Correlation Analysis
    print("\n" + "=" * 60)
    print("3. Correlation Analysis")
    print("=" * 60)
    
    # Compute correlation matrix
    corr_matrix = compute_correlations(df)
    
    # Run complete EDA pipeline
    run_eda_pipeline(df, figures_dir='figures')
    
    # Save to Excel
    print("\n" + "=" * 60)
    print("5. Saving Results to Excel")
    print("=" * 60)
    save_eda_to_excel(df, corr_matrix, 'rapor/01_eda_sonuclari.xlsx')
    
    # 4. Additional Scatter Plots
    print("\n" + "=" * 60)
    print("4. Generating Additional Scatter Plots")
    print("=" * 60)
    
    # Light vs Temperature
    if 'light' in df.columns and 'temperature' in df.columns:
        plot_scatter(df, 'light', 'temperature',
                    save_path='figures/light_vs_temperature.png',
                    title='Light vs Temperature')
    
    # Temperature vs Humidity
    if 'temperature' in df.columns and 'humidity' in df.columns:
        plot_scatter(df, 'temperature', 'humidity',
                    save_path='figures/temperature_vs_humidity.png',
                    title='Temperature vs Humidity')
    
    # CO2 vs Temperature
    if 'co2' in df.columns and 'temperature' in df.columns:
        plot_scatter(df, 'co2', 'temperature',
                    save_path='figures/co2_vs_temperature.png',
                    title='CO₂ vs Temperature')
    
    # pH vs EC
    if 'pH' in df.columns and 'EC' in df.columns:
        plot_scatter(df, 'pH', 'EC',
                    save_path='figures/ph_vs_ec.png',
                    title='pH vs EC')
    
    print("\n" + "=" * 60)
    print("Exploratory Data Analysis Complete!")
    print("=" * 60)
    print(f"All figures saved to: figures/")
    print(f"Excel report saved to: rapor/01_eda_sonuclari.xlsx")


if __name__ == '__main__':
    main()

