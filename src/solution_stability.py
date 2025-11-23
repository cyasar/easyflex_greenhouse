"""
Solution Stability Analysis Module.

This module quantifies the stability of optimal spin configurations under different
random seeds for the simulated annealing optimizer. Uses Hamming distance to measure
solution similarity across multiple optimization runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dimod
from typing import Dict, List
import os

from build_hamiltonian import (
    build_hamiltonian_from_data,
    SPIN_VARIABLES
)
from optimise_sa import optimize_with_sa


def run_with_seed(bqm: dimod.BinaryQuadraticModel,
                  seed: int,
                  num_reads: int = 500) -> Dict:
    """
    Run simulated annealing optimization with a specific random seed.
    
    Parameters:
    -----------
    bqm : dimod.BinaryQuadraticModel
        Ising Hamiltonian to optimize
    seed : int
        Random seed for reproducibility
    num_reads : int
        Number of SA reads
        
    Returns:
    --------
    dict
        Dictionary with keys: 'seed', 'sample', 'energy'
        where 'sample' is {spin_name: ±1}
    """
    best_sample, best_energy, metadata = optimize_with_sa(
        bqm, num_reads=num_reads, seed=seed
    )
    
    return {
        'seed': seed,
        'sample': best_sample,
        'energy': best_energy
    }


def spin_vector_to_array(sample: Dict[str, int],
                         spin_names: List[str] = None) -> np.ndarray:
    """
    Convert spin configuration dictionary to numpy array.
    
    Parameters:
    -----------
    sample : dict
        Spin configuration {spin_name: spin_value}
    spin_names : list of str, optional
        Ordered list of spin names. If None, uses SPIN_VARIABLES.
        
    Returns:
    --------
    np.ndarray
        Array of spin values in order
    """
    if spin_names is None:
        spin_names = SPIN_VARIABLES
    
    return np.array([sample.get(name, 0) for name in spin_names])


def hamming_distance(sample1: Dict[str, int],
                     sample2: Dict[str, int],
                     spin_names: List[str] = None) -> float:
    """
    Compute normalized Hamming distance between two spin configurations.
    
    Hamming distance = fraction of spins that differ.
    Range: [0, 1] where 0 = identical, 1 = all spins differ.
    
    Parameters:
    -----------
    sample1 : dict
        First spin configuration
    sample2 : dict
        Second spin configuration
    spin_names : list of str, optional
        Ordered list of spin names
        
    Returns:
    --------
    float
        Normalized Hamming distance (0-1)
    """
    if spin_names is None:
        spin_names = SPIN_VARIABLES
    
    vec1 = spin_vector_to_array(sample1, spin_names)
    vec2 = spin_vector_to_array(sample2, spin_names)
    
    # Count differences
    n_different = np.sum(vec1 != vec2)
    n_total = len(spin_names)
    
    return n_different / n_total


def compute_hamming_matrix(results: List[Dict],
                          spin_names: List[str] = None) -> np.ndarray:
    """
    Compute pairwise Hamming distance matrix for all solutions.
    
    Parameters:
    -----------
    results : list of dict
        List of result dictionaries from run_with_seed()
    spin_names : list of str, optional
        Ordered list of spin names
        
    Returns:
    --------
    np.ndarray
        Symmetric matrix of Hamming distances, shape (n_results, n_results)
    """
    n = len(results)
    hamming_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                hamming_matrix[i, j] = 0.0
            else:
                dist = hamming_distance(
                    results[i]['sample'],
                    results[j]['sample'],
                    spin_names
                )
                hamming_matrix[i, j] = dist
                hamming_matrix[j, i] = dist  # Symmetric
    
    return hamming_matrix


def compute_average_hamming_distances(hamming_matrix: np.ndarray) -> np.ndarray:
    """
    Compute average Hamming distance from each solution to all others.
    
    Parameters:
    -----------
    hamming_matrix : np.ndarray
        Pairwise Hamming distance matrix
        
    Returns:
    --------
    np.ndarray
        Array of average distances for each solution
    """
    n = hamming_matrix.shape[0]
    avg_distances = np.zeros(n)
    
    for i in range(n):
        # Average over all other solutions (exclude self)
        distances_to_others = np.concatenate([
            hamming_matrix[i, :i],
            hamming_matrix[i, i+1:]
        ])
        avg_distances[i] = np.mean(distances_to_others)
    
    return avg_distances


def plot_hamming_heatmap(hamming_matrix: np.ndarray,
                         seeds: List[int],
                         save_path: str = 'figures/solution_stability_hamming.png'):
    """
    Plot Hamming distance matrix as a heatmap.
    
    Parameters:
    -----------
    hamming_matrix : np.ndarray
        Pairwise Hamming distance matrix
    seeds : list of int
        List of seeds corresponding to matrix rows/columns
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    im = ax.imshow(hamming_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Hamming Distance', fontsize=12, fontweight='bold')
    
    # Set ticks and labels
    ax.set_xticks(range(len(seeds)))
    ax.set_yticks(range(len(seeds)))
    ax.set_xticklabels(seeds, fontsize=10)
    ax.set_yticklabels(seeds, fontsize=10)
    
    # Add text annotations
    for i in range(len(seeds)):
        for j in range(len(seeds)):
            text = ax.text(j, i, f'{hamming_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # Labels
    ax.set_xlabel('Seed', fontsize=12, fontweight='bold')
    ax.set_ylabel('Seed', fontsize=12, fontweight='bold')
    ax.set_title('Solution Stability: Pairwise Hamming Distance Matrix',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Hamming distance heatmap to {save_path}")
    plt.close()


def main():
    """
    Main function: analyze solution stability across multiple random seeds.
    """
    print("=" * 80)
    print("SOLUTION STABILITY ANALYSIS")
    print("=" * 80)
    
    # Build Hamiltonian (using default parameters)
    print("\n1. Building Ising Hamiltonian...")
    sensor_data = {
        'temperature': 20.0,
        'humidity': 60.0,
        'co2': 400.0
    }
    
    bqm, J, h = build_hamiltonian_from_data(
        sensor_data=sensor_data,
        n_spins=8,
        interaction_strength=1.0,
        field_strength=1.0,
        co2_penalty=0.0
    )
    print(f"   Hamiltonian built with {len(bqm.variables)} variables")
    
    # Define seeds
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"\n2. Running optimization with {len(seeds)} different seeds...")
    print("-" * 80)
    
    # Run optimization for each seed
    results = []
    for i, seed in enumerate(seeds, 1):
        print(f"   [{i}/{len(seeds)}] Seed {seed}...", end=' ', flush=True)
        result = run_with_seed(bqm, seed, num_reads=500)
        results.append(result)
        print(f"Energy: {result['energy']:.4f}")
    
    # Compute Hamming distance matrix
    print("\n3. Computing pairwise Hamming distances...")
    hamming_matrix = compute_hamming_matrix(results)
    avg_distances = compute_average_hamming_distances(hamming_matrix)
    
    # Build summary DataFrame
    summary_data = {
        'seed': [r['seed'] for r in results],
        'energy': [r['energy'] for r in results],
        'avg_hamming_distance': avg_distances
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("STABILITY SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    
    print(f"\nStatistics:")
    print(f"  Mean energy: {summary_df['energy'].mean():.4f} ± {summary_df['energy'].std():.4f}")
    print(f"  Mean Hamming distance: {summary_df['avg_hamming_distance'].mean():.4f} ± "
          f"{summary_df['avg_hamming_distance'].std():.4f}")
    print(f"  Energy range: [{summary_df['energy'].min():.4f}, {summary_df['energy'].max():.4f}]")
    print(f"  Hamming distance range: [{summary_df['avg_hamming_distance'].min():.4f}, "
          f"{summary_df['avg_hamming_distance'].max():.4f}]")
    
    # Plot Hamming distance heatmap
    print("\n4. Generating Hamming distance heatmap...")
    plot_hamming_heatmap(hamming_matrix, seeds)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: figures/solution_stability_hamming.png")
    print()


if __name__ == '__main__':
    main()

