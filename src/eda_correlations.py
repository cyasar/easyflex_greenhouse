"""
Exploratory Data Analysis module for greenhouse optimization project.

This module computes Pearson correlations between sensor variables and
produces scatter plots for visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple
import os


def compute_correlations(df: pd.DataFrame, 
                        variables: List[str] = None) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix for sensor variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with sensor columns
    variables : list of str, optional
        List of variable names to correlate. If None, uses default sensor list.
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    if variables is None:
        variables = ['temperature', 'humidity', 'light', 'co2', 'pH', 'EC']
    
    # Filter to available columns
    available_vars = [v for v in variables if v in df.columns]
    
    if len(available_vars) < 2:
        raise ValueError("Need at least 2 variables for correlation analysis")
    
    corr_matrix = df[available_vars].corr(method='pearson')
    return corr_matrix


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, 
                             save_path: str = 'figures/correlation_heatmap.png'):
    """
    Plot correlation matrix as a heatmap.
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    save_path : str
        Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Pearson Correlation Matrix - Greenhouse Sensors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation heatmap to {save_path}")
    plt.close()


def plot_scatter(df: pd.DataFrame, x_var: str, y_var: str,
                save_path: str = None, title: str = None):
    """
    Create a scatter plot between two variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    x_var : str
        X-axis variable name
    y_var : str
        Y-axis variable name
    save_path : str, optional
        Path to save the figure. If None, displays plot.
    title : str, optional
        Plot title. If None, auto-generates from variable names.
    """
    if x_var not in df.columns or y_var not in df.columns:
        raise ValueError(f"Variables {x_var} or {y_var} not found in dataframe")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_var], df[y_var], alpha=0.5, s=20)
    plt.xlabel(x_var.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_var.replace('_', ' ').title(), fontsize=12)
    
    if title is None:
        title = f'{y_var.replace("_", " ").title()} vs {x_var.replace("_", " ").title()}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to {save_path}")
        plt.close()
    else:
        plt.show()


def plot_light_vs_temperature(df: pd.DataFrame, 
                              save_path: str = 'figures/light_vs_temperature.png'):
    """
    Plot Light vs Temperature scatter plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_path : str
        Path to save the figure
    """
    plot_scatter(df, 'light', 'temperature', 
                save_path=save_path, 
                title='Light vs Temperature')


def plot_temperature_vs_humidity(df: pd.DataFrame,
                                save_path: str = 'figures/temperature_vs_humidity.png'):
    """
    Plot Temperature vs Humidity scatter plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    save_path : str
        Path to save the figure
    """
    plot_scatter(df, 'temperature', 'humidity',
                save_path=save_path,
                title='Temperature vs Humidity')


def run_eda_pipeline(df: pd.DataFrame, figures_dir: str = 'figures'):
    """
    Run complete EDA pipeline: correlations and key scatter plots.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    figures_dir : str
        Directory to save figures
    """
    print("Running EDA pipeline...")
    
    # Compute correlations
    corr_matrix = compute_correlations(df)
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Plot correlation heatmap
    plot_correlation_heatmap(corr_matrix, 
                            save_path=f'{figures_dir}/correlation_heatmap.png')
    
    # Plot key scatter plots
    if 'light' in df.columns and 'temperature' in df.columns:
        plot_light_vs_temperature(df, 
                                 save_path=f'{figures_dir}/light_vs_temperature.png')
    
    if 'temperature' in df.columns and 'humidity' in df.columns:
        plot_temperature_vs_humidity(df,
                                    save_path=f'{figures_dir}/temperature_vs_humidity.png')
    
    print(f"\nEDA pipeline complete. Figures saved to {figures_dir}/")


if __name__ == '__main__':
    # Example usage
    from load_data import preprocess_greenhouse_data
    
    df, _ = preprocess_greenhouse_data()
    run_eda_pipeline(df)

