"""
CO₂ Ablation Study module.

This module fixes the CO₂ spin to -1 (forced OFF) and recomputes the
Hamiltonian energy to compare against the normal optimization case.
"""

import numpy as np
from typing import Dict, Tuple
from build_hamiltonian import build_hamiltonian_from_data, compute_hamiltonian_energy
from optimise_sa import run_sa_optimization
import json
import os


def run_co2_ablation_study(sensor_data: Dict[str, float] = None,
                           num_reads: int = 1000,
                           co2_penalty: float = 0.0) -> Dict:
    """
    Run CO₂ ablation study: compare optimization with CO₂ enabled vs disabled.
    
    Parameters:
    -----------
    sensor_data : dict, optional
        Dictionary of sensor readings
    num_reads : int
        Number of optimization reads
    co2_penalty : float
        CO₂ penalty weight for normal case
        
    Returns:
    --------
    dict
        Results dictionary with normal and ablation cases
    """
    if sensor_data is None:
        sensor_data = {'temperature': 20.0, 'humidity': 60.0, 'co2': 400.0}
    
    print("Running CO₂ Ablation Study...")
    print("=" * 50)
    
    # Case 1: Normal optimization (CO₂ can be ON or OFF)
    print("\n--- Case 1: Normal Optimization (CO₂ enabled) ---")
    bqm_normal, J_normal, h_normal = build_hamiltonian_from_data(
        sensor_data=sensor_data,
        co2_penalty=co2_penalty
    )
    
    best_sample_normal, best_energy_normal, metadata_normal = run_sa_optimization(
        bqm_normal, num_reads=num_reads, verbose=False
    )
    
    co2_state_normal = 'ON' if best_sample_normal.get('CO2', -1) == 1 else 'OFF'
    print(f"Best energy: {best_energy_normal:.4f}")
    print(f"CO₂ state: {co2_state_normal}")
    
    # Case 2: Ablation (CO₂ forced OFF)
    print("\n--- Case 2: Ablation (CO₂ forced OFF) ---")
    bqm_ablation, J_ablation, h_ablation = build_hamiltonian_from_data(
        sensor_data=sensor_data,
        co2_penalty=co2_penalty
    )
    
    # Fix CO₂ to -1 (OFF) by adding a large penalty for +1 state
    # This effectively forces CO₂ spin to -1
    if 'CO2' in bqm_ablation.variables:
        # Add large linear bias to force CO₂ to -1
        large_penalty = 1000.0
        bqm_ablation.add_variable('CO2', large_penalty)
    
    # Optimize with CO₂ constraint
    best_sample_ablation, best_energy_ablation, metadata_ablation = run_sa_optimization(
        bqm_ablation, num_reads=num_reads, verbose=False
    )
    
    # Verify CO₂ is OFF
    co2_state_ablation = 'ON' if best_sample_ablation.get('CO2', -1) == 1 else 'OFF'
    print(f"Best energy: {best_energy_ablation:.4f}")
    print(f"CO₂ state: {co2_state_ablation} (should be OFF)")
    
    # Compute energy difference
    energy_difference = best_energy_ablation - best_energy_normal
    energy_ratio = best_energy_ablation / best_energy_normal if best_energy_normal != 0 else np.inf
    
    print("\n" + "=" * 50)
    print("Ablation Results:")
    print(f"  Normal case energy:    {best_energy_normal:.4f}")
    print(f"  Ablation case energy:  {best_energy_ablation:.4f}")
    print(f"  Energy difference:     {energy_difference:.4f}")
    print(f"  Energy ratio:          {energy_ratio:.4f}")
    
    results = {
        'normal': {
            'energy': float(best_energy_normal),
            'configuration': {k: int(v) for k, v in best_sample_normal.items()},
            'co2_state': co2_state_normal,
            'metadata': {
                'mean_energy': float(metadata_normal['mean_energy']),
                'std_energy': float(metadata_normal['std_energy']),
                'elapsed_time': float(metadata_normal['elapsed_time'])
            }
        },
        'ablation': {
            'energy': float(best_energy_ablation),
            'configuration': {k: int(v) for k, v in best_sample_ablation.items()},
            'co2_state': co2_state_ablation,
            'metadata': {
                'mean_energy': float(metadata_ablation['mean_energy']),
                'std_energy': float(metadata_ablation['std_energy']),
                'elapsed_time': float(metadata_ablation['elapsed_time'])
            }
        },
        'comparison': {
            'energy_difference': float(energy_difference),
            'energy_ratio': float(energy_ratio),
            'co2_penalty': float(co2_penalty)
        }
    }
    
    return results


def save_ablation_results(results: Dict, filepath: str = 'results/co2_ablation.json'):
    """
    Save ablation study results to JSON file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_co2_ablation_study
    filepath : str
        Path to save JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filepath}")


def load_ablation_results(filepath: str = 'results/co2_ablation.json') -> Dict:
    """
    Load ablation study results from JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to JSON file
        
    Returns:
    --------
    dict
        Results dictionary
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def plot_ablation_comparison(results: Dict, save_path: str = 'figures/co2_ablation.png'):
    """
    Plot ablation study comparison: normal vs ablation energy.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_co2_ablation_study
    save_path : str
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    normal_energy = results['normal']['energy']
    ablation_energy = results['ablation']['energy']
    
    plt.figure(figsize=(8, 6))
    cases = ['Normal\n(CO₂ enabled)', 'Ablation\n(CO₂ forced OFF)']
    energies = [normal_energy, ablation_energy]
    colors = ['#2ecc71', '#e74c3c']
    
    bars = plt.bar(cases, energies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.ylabel('Hamiltonian Energy', fontsize=12)
    plt.title('CO₂ Ablation Study: Energy Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ablation comparison to {save_path}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    print("Running CO₂ Ablation Study...")
    
    results = run_co2_ablation_study(
        sensor_data={'temperature': 20.0, 'humidity': 60.0},
        num_reads=500,
        co2_penalty=0.0
    )
    
    # Save results
    save_ablation_results(results)
    
    # Plot
    plot_ablation_comparison(results)
    
    print("\nAblation study complete!")

