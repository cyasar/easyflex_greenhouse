"""
CO₂ Sensitivity Analysis module.

This module re-runs optimization with different CO₂ penalty weights (lambda)
to analyze sensitivity of the optimization to CO₂ usage costs.
"""

import numpy as np
import json
from typing import Dict, List, Tuple
from build_hamiltonian import build_hamiltonian_from_data
from optimise_sa import run_sa_optimization
import os


def run_co2_sensitivity_analysis(sensor_data: Dict[str, float] = None,
                                 lambda_values: List[float] = None,
                                 num_reads: int = 1000,
                                 use_quantum: bool = False,
                                 quantum_token: str = None) -> Dict:
    """
    Run CO₂ sensitivity analysis by varying penalty weight lambda.
    
    Parameters:
    -----------
    sensor_data : dict, optional
        Dictionary of sensor readings
    lambda_values : list of float, optional
        List of CO₂ penalty weights to test. Default: [0.0, 0.25, 0.5, 1.0]
    num_reads : int
        Number of optimization reads per lambda value
    use_quantum : bool
        Whether to use quantum annealing (requires credentials)
    quantum_token : str, optional
        D-Wave API token if using quantum annealing
        
    Returns:
    --------
    dict
        Results dictionary with keys: lambda_values, energies, configurations, metadata
    """
    if lambda_values is None:
        lambda_values = [0.0, 0.25, 0.5, 1.0]
    
    if sensor_data is None:
        sensor_data = {'temperature': 20.0, 'humidity': 60.0, 'co2': 400.0}
    
    results = {
        'lambda_values': lambda_values,
        'energies': [],
        'configurations': [],
        'metadata': []
    }
    
    print("Running CO₂ Sensitivity Analysis...")
    print(f"Lambda values: {lambda_values}")
    print(f"Optimization method: {'Quantum Annealing' if use_quantum else 'Simulated Annealing'}")
    
    for lam in lambda_values:
        print(f"\n--- Lambda = {lam} ---")
        
        # Build Hamiltonian with CO₂ penalty
        bqm, J, h = build_hamiltonian_from_data(
            sensor_data=sensor_data,
            co2_penalty=lam
        )
        
        # Optimize
        if use_quantum:
            try:
                from optimise_qanneal import run_quantum_optimization
                best_sample, best_energy, metadata = run_quantum_optimization(
                    bqm, num_reads=num_reads, verbose=False, token=quantum_token
                )
            except Exception as e:
                print(f"Quantum optimization failed: {e}")
                print("Falling back to Simulated Annealing...")
                best_sample, best_energy, metadata = run_sa_optimization(
                    bqm, num_reads=num_reads, verbose=False
                )
        else:
            best_sample, best_energy, metadata = run_sa_optimization(
                bqm, num_reads=num_reads, verbose=False
            )
        
        # Store results
        results['energies'].append(float(best_energy))
        results['configurations'].append({k: int(v) for k, v in best_sample.items()})
        results['metadata'].append({
            'lambda': float(lam),
            'best_energy': float(best_energy),
            'mean_energy': float(metadata['mean_energy']),
            'std_energy': float(metadata['std_energy']),
            'elapsed_time': float(metadata['elapsed_time'])
        })
        
        # Print CO₂ state
        co2_state = 'ON' if best_sample.get('CO2', -1) == 1 else 'OFF'
        print(f"  Best energy: {best_energy:.4f}")
        print(f"  CO₂ state: {co2_state}")
    
    return results


def save_sensitivity_results(results: Dict, filepath: str = 'results/co2_sensitivity.json'):
    """
    Save sensitivity analysis results to JSON file.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_co2_sensitivity_analysis
    filepath : str
        Path to save JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filepath}")


def load_sensitivity_results(filepath: str = 'results/co2_sensitivity.json') -> Dict:
    """
    Load sensitivity analysis results from JSON file.
    
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


def plot_sensitivity_curve(results: Dict, save_path: str = 'figures/co2_sensitivity.png'):
    """
    Plot CO₂ sensitivity curve: energy vs lambda.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_co2_sensitivity_analysis
    save_path : str
        Path to save figure
    """
    import matplotlib.pyplot as plt
    
    lambda_values = results['lambda_values']
    energies = results['energies']
    
    plt.figure(figsize=(8, 6))
    plt.plot(lambda_values, energies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('CO₂ Penalty Weight (λ)', fontsize=12)
    plt.ylabel('Hamiltonian Energy', fontsize=12)
    plt.title('CO₂ Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved sensitivity curve to {save_path}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    print("Running CO₂ Sensitivity Analysis...")
    
    results = run_co2_sensitivity_analysis(
        sensor_data={'temperature': 20.0, 'humidity': 60.0},
        lambda_values=[0.0, 0.25, 0.5, 1.0],
        num_reads=500
    )
    
    # Save results
    save_sensitivity_results(results)
    
    # Plot
    plot_sensitivity_curve(results)
    
    print("\nSensitivity analysis complete!")

