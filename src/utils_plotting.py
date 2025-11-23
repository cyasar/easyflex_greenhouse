"""
Central plotting utilities for greenhouse optimization project.

This module provides reusable plotting functions for figures used in
the Jupyter notebooks and analysis scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import os


# Set default style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


def setup_plot_style(figsize: Tuple[int, int] = (10, 6),
                    dpi: int = 300,
                    style: str = 'seaborn-v0_8-darkgrid'):
    """
    Setup matplotlib plotting style.
    
    Parameters:
    -----------
    figsize : tuple
        Default figure size (width, height)
    dpi : int
        Default DPI for saved figures
    style : str
        Matplotlib style name
    """
    if style in plt.style.available:
        plt.style.use(style)
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def save_figure(fig, filepath: str, dpi: int = 300, bbox_inches: str = 'tight'):
    """
    Save figure to file with consistent settings.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object to save
    filepath : str
        Path to save figure
    dpi : int
        Resolution for saved figure
    bbox_inches : str
        Bounding box setting
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Saved figure to {filepath}")


def plot_time_series(df: pd.DataFrame,
                    columns: List[str],
                    title: str = "Time Series Data",
                    save_path: Optional[str] = None):
    """
    Plot time series for multiple columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with time series data
    columns : list of str
        Column names to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col in columns:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col.replace('_', ' ').title(), alpha=0.7)
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    plt.close()


def plot_energy_landscape(energies: List[float],
                         labels: Optional[List[str]] = None,
                         title: str = "Energy Landscape",
                         save_path: Optional[str] = None):
    """
    Plot energy landscape from optimization runs.
    
    Parameters:
    -----------
    energies : list of float
        List of energy values
    labels : list of str, optional
        Labels for each energy value
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(energies))
    ax.plot(x, energies, 'o-', linewidth=2, markersize=6, alpha=0.7)
    
    if labels:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    plt.close()


def plot_spin_configuration(spin_config: Dict[str, int],
                           title: str = "Spin Configuration",
                           save_path: Optional[str] = None):
    """
    Plot spin configuration as a bar chart.
    
    Parameters:
    -----------
    spin_config : dict
        Spin configuration {name: value} where value ∈ {-1, +1}
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = list(spin_config.keys())
    values = list(spin_config.values())
    colors = ['#e74c3c' if v == -1 else '#2ecc71' for v in values]
    labels = ['OFF' if v == -1 else 'ON' for v in values]
    
    bars = ax.bar(names, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add labels
    for bar, label in zip(bars, labels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label, ha='center', va='bottom' if height > 0 else 'top',
               fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Spin Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    plt.close()


def plot_interaction_matrix(J: np.ndarray,
                           spin_names: Optional[List[str]] = None,
                           title: str = "Interaction Matrix",
                           save_path: Optional[str] = None):
    """
    Plot interaction matrix as a heatmap.
    
    Parameters:
    -----------
    J : np.ndarray
        Interaction matrix
    spin_names : list of str, optional
        Names for spin variables
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if spin_names is None:
        spin_names = [f'Spin {i}' for i in range(J.shape[0])]
    
    sns.heatmap(J, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, linewidths=1, cbar_kws={"shrink": 0.8},
               xticklabels=spin_names, yticklabels=spin_names, ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    plt.close()


def plot_external_field(h: np.ndarray,
                       spin_names: Optional[List[str]] = None,
                       title: str = "External Field",
                       save_path: Optional[str] = None):
    """
    Plot external field vector as a bar chart.
    
    Parameters:
    -----------
    h : np.ndarray
        External field vector
    spin_names : list of str, optional
        Names for spin variables
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if spin_names is None:
        spin_names = [f'Spin {i}' for i in range(len(h))]
    
    colors = ['#3498db' if v >= 0 else '#e74c3c' for v in h]
    bars = ax.bar(spin_names, h, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Field Strength', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    plt.close()


def plot_comparison_bar(categories: List[str],
                       values1: List[float],
                       values2: List[float],
                       label1: str = "Case 1",
                       label2: str = "Case 2",
                       title: str = "Comparison",
                       save_path: Optional[str] = None):
    """
    Plot side-by-side bar comparison.
    
    Parameters:
    -----------
    categories : list of str
        Category names
    values1 : list of float
        Values for first case
    values2 : list of float
        Values for second case
    label1 : str
        Label for first case
    label2 : str
        Label for second case
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, values1, width, label=label1, alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, values2, width, label=label2, alpha=0.7, edgecolor='black')
    
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    plt.close()


# Initialize default style
setup_plot_style()

