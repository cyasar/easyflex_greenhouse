"""
Noise Robustness Analysis Module.

This module assesses the robustness of optimal control decisions (spin configurations)
to sensor noise. Perturbs environmental statistics and analyzes solution stability
across multiple noisy scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dimod
from typing import Dict, List, Tuple
import os
from collections import Counter

from load_data import load_greenhouse_data
from build_hamiltonian import build_hamiltonian_from_data, SPIN_VARIABLES
from optimise_sa import optimize_with_sa


# Noise levels (standard deviations) for each sensor
NOISE_LEVELS = {
    'temperature': 0.5,  # °C
    'humidity': 2.0,      # %
    'co2': 20.0,         # ppm
    'light': 50.0         # lux (if available)
}


def load_greenhouse_statistics(df: pd.DataFrame,
                              greenhouse: str = 'A',
                              day_window: int = 7) -> Dict[str, float]:
    """
    Load and compute baseline statistics for a specific greenhouse and time window.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Greenhouse data dataframe
    greenhouse : str
        Greenhouse identifier (e.g., 'A', 'B', 'C', 'D')
    day_window : int
        Number of days to include in the window (default: 7)
        
    Returns:
    --------
    dict
        Dictionary with mean statistics: 'temperature', 'humidity', 'co2', etc.
    """
    # Filter by greenhouse
    if 'greenhouse' in df.columns:
        gh_data = df[df['greenhouse'] == greenhouse].copy()
    else:
        # If no greenhouse column, use all data
        gh_data = df.copy()
        print(f"Warning: No 'greenhouse' column found. Using all data.")
    
    # If timestamp exists, filter by recent days
    if 'timestamp' in gh_data.columns:
        if not pd.api.types.is_datetime64_any_dtype(gh_data['timestamp']):
            gh_data['timestamp'] = pd.to_datetime(gh_data['timestamp'])
        
        # Get most recent day_window days
        max_date = gh_data['timestamp'].max()
        min_date = max_date - pd.Timedelta(days=day_window)
        gh_data = gh_data[gh_data['timestamp'] >= min_date]
    
    # Compute means
    stats = {}
    for var in ['temperature', 'humidity', 'co2', 'light']:
        if var in gh_data.columns:
            stats[var] = gh_data[var].mean()
        else:
            # Use default values if column missing
            defaults = {
                'temperature': 20.0,
                'humidity': 60.0,
                'co2': 400.0,
                'light': 500.0
            }
            stats[var] = defaults.get(var, 0.0)
            print(f"Warning: '{var}' not found. Using default: {stats[var]}")
    
    return stats


def add_gaussian_noise(stats: Dict[str, float],
                      noise_levels: Dict[str, float] = None,
                      seed: int = None) -> Dict[str, float]:
    """
    Add Gaussian noise to statistics.
    
    Parameters:
    -----------
    stats : dict
        Baseline statistics dictionary
    noise_levels : dict, optional
        Standard deviations for each variable. If None, uses default NOISE_LEVELS.
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    dict
        Perturbed statistics dictionary
    """
    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    
    if seed is not None:
        np.random.seed(seed)
    
    perturbed = {}
    for var, value in stats.items():
        sigma = noise_levels.get(var, 0.1 * abs(value))  # Default: 10% of value
        noise = np.random.normal(0, sigma)
        perturbed[var] = value + noise
    
    return perturbed


def build_hamiltonian_from_stats(stats: Dict[str, float],
                                n_spins: int = 8,
                                interaction_strength: float = 1.0,
                                field_strength: float = 1.0) -> dimod.BinaryQuadraticModel:
    """
    Build Hamiltonian with external field adjusted based on environmental statistics.
    
    The external field coefficients are adjusted based on sensor readings:
    - Low temperature → bias heater towards ON
    - High temperature → bias fan towards ON
    - Low CO2 → bias CO2 injection towards ON
    - etc.
    
    Parameters:
    -----------
    stats : dict
        Environmental statistics dictionary
    n_spins : int
        Number of spin variables
    interaction_strength : float
        Base interaction strength
    field_strength : float
        Base external field strength
        
    Returns:
    --------
    dimod.BinaryQuadraticModel
        Ising Hamiltonian with adjusted external field
    """
    # Build Hamiltonian with sensor data
    bqm, J, h = build_hamiltonian_from_data(
        sensor_data=stats,
        n_spins=n_spins,
        interaction_strength=interaction_strength,
        field_strength=field_strength,
        co2_penalty=0.0
    )
    
    return bqm


def spin_config_to_tuple(sample: Dict[str, int],
                        spin_names: List[str] = None) -> Tuple:
    """
    Convert spin configuration to tuple for hashing.
    
    Parameters:
    -----------
    sample : dict
        Spin configuration
    spin_names : list of str, optional
        Ordered list of spin names
        
    Returns:
    --------
    tuple
        Tuple of spin values in order
    """
    if spin_names is None:
        spin_names = SPIN_VARIABLES
    
    return tuple(sample.get(name, 0) for name in spin_names)


def compute_spin_stability(all_samples: List[Dict[str, int]],
                          spin_names: List[str] = None) -> pd.DataFrame:
    """
    Compute stability (frequency of mode pattern) for each spin.
    
    Parameters:
    -----------
    all_samples : list of dict
        List of spin configurations from noisy runs
    spin_names : list of str, optional
        Ordered list of spin names
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with columns: 'spin_name', 'stability'
    """
    if spin_names is None:
        spin_names = SPIN_VARIABLES
    
    n_runs = len(all_samples)
    
    # Find most frequent pattern (mode)
    config_tuples = [spin_config_to_tuple(sample, spin_names) for sample in all_samples]
    counter = Counter(config_tuples)
    mode_config_tuple = counter.most_common(1)[0][0]
    mode_config = {name: mode_config_tuple[i] for i, name in enumerate(spin_names)}
    
    # Compute stability for each spin
    stability_data = []
    for spin_name in spin_names:
        # Count how many times this spin matches the mode
        matches = sum(1 for sample in all_samples 
                     if sample.get(spin_name, 0) == mode_config[spin_name])
        stability = matches / n_runs
        stability_data.append({
            'spin_name': spin_name,
            'stability': stability,
            'mode_value': mode_config[spin_name],
            'mode_state': 'ON' if mode_config[spin_name] == 1 else 'OFF'
        })
    
    return pd.DataFrame(stability_data)


def plot_spin_stability(stability_df: pd.DataFrame,
                       save_path: str = 'figures/noise_robustness_spin_stability.png'):
    """
    Plot bar chart of spin stability.
    
    Parameters:
    -----------
    stability_df : pd.DataFrame
        Stability dataframe from compute_spin_stability()
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by stability for better visualization
    sorted_df = stability_df.sort_values('stability', ascending=True)
    
    # Color bars based on stability
    colors = ['#e74c3c' if s < 0.7 else '#f39c12' if s < 0.9 else '#2ecc71' 
              for s in sorted_df['stability']]
    
    bars = ax.barh(sorted_df['spin_name'], sorted_df['stability'],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (bar, stability) in enumerate(zip(bars, sorted_df['stability'])):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
               f'{stability:.2f}',
               ha='left' if width < 0.1 else 'right',
               va='center', fontsize=10, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Stability (Fraction of Runs Matching Mode)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Spin Variable', fontsize=12, fontweight='bold')
    ax.set_title('Spin Stability Under Sensor Noise',
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label='Low (< 0.7)'),
        Patch(facecolor='#f39c12', alpha=0.7, label='Medium (0.7-0.9)'),
        Patch(facecolor='#2ecc71', alpha=0.7, label='High (≥ 0.9)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved spin stability plot to {save_path}")
    plt.close()


def main():
    """
    Main function: analyze robustness to sensor noise.
    """
    print("=" * 80)
    print("NOISE ROBUSTNESS ANALYSIS")
    print("=" * 80)
    
    # 1. Load data
    print("\n1. Loading greenhouse data...")
    filepath = 'data/greenhouse_data.xlsx'
    
    # Try multiple possible paths
    possible_paths = [
        'data/greenhouse_data.xlsx',
        '../data/greenhouse_data.xlsx',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'greenhouse_data.xlsx')
    ]
    
    df = None
    for path in possible_paths:
        try:
            df = load_greenhouse_data(path)
            print(f"   Loaded {len(df)} rows from {path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        print("   Warning: Could not load data file. Using default statistics.")
        baseline_stats = {
            'temperature': 20.0,
            'humidity': 60.0,
            'co2': 400.0,
            'light': 500.0
        }
    else:
        # 2. Compute baseline statistics
        print("\n2. Computing baseline statistics...")
        greenhouse = 'A'
        baseline_stats = load_greenhouse_statistics(df, greenhouse=greenhouse, day_window=7)
        print(f"   Greenhouse: {greenhouse}")
        print(f"   Baseline statistics:")
        for var, value in baseline_stats.items():
            print(f"     {var}: {value:.2f}")
    
    # 3. Generate noisy scenarios
    print("\n3. Generating noisy scenarios...")
    n_scenarios = 50
    print(f"   Number of scenarios: {n_scenarios}")
    print(f"   Noise levels: {NOISE_LEVELS}")
    
    noisy_stats_list = []
    for i in range(n_scenarios):
        noisy_stats = add_gaussian_noise(baseline_stats, seed=i)
        noisy_stats_list.append(noisy_stats)
    
    # 4. Run optimization for each noisy scenario
    print("\n4. Running optimization for each noisy scenario...")
    print("-" * 80)
    all_samples = []
    
    for i in range(n_scenarios):
        if (i + 1) % 10 == 0:
            print(f"   [{i+1}/{n_scenarios}]...", end=' ', flush=True)
        
        # Build Hamiltonian with noisy stats
        bqm = build_hamiltonian_from_stats(noisy_stats_list[i])
        
        # Optimize
        best_sample, best_energy, metadata = optimize_with_sa(
            bqm, num_reads=500, seed=42
        )
        all_samples.append(best_sample)
        
        if (i + 1) % 10 == 0:
            print(f"Done")
    
    print(f"   Completed {n_scenarios} optimizations")
    
    # 5. Compute spin stability
    print("\n5. Computing spin stability...")
    stability_df = compute_spin_stability(all_samples)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("SPIN STABILITY SUMMARY")
    print("=" * 80)
    print(stability_df.to_string(index=False))
    
    print(f"\nStatistics:")
    print(f"  Mean stability: {stability_df['stability'].mean():.4f} ± "
          f"{stability_df['stability'].std():.4f}")
    print(f"  Min stability: {stability_df['stability'].min():.4f}")
    print(f"  Max stability: {stability_df['stability'].max():.4f}")
    print(f"  Spins with stability ≥ 0.9: "
          f"{(stability_df['stability'] >= 0.9).sum()}/{len(stability_df)}")
    
    # 6. Plot results
    print("\n6. Generating spin stability plot...")
    plot_spin_stability(stability_df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: figures/noise_robustness_spin_stability.png")
    print()


if __name__ == '__main__':
    main()

