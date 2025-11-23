"""
Classical Simulated Annealing optimization module.

This module implements classical optimization using neal.SimulatedAnnealingSampler
for the greenhouse Ising Hamiltonian.
"""

import numpy as np
import dimod
from neal import SimulatedAnnealingSampler
from typing import Dict, Tuple, Optional
import time


def optimize_with_sa(bqm: dimod.BinaryQuadraticModel,
                     num_reads: int = 1000,
                     num_sweeps: int = 1000,
                     beta_range: Tuple[float, float] = (0.1, 50.0),
                     seed: Optional[int] = None) -> Tuple[Dict[str, int], float, Dict]:
    """
    Optimize Ising Hamiltonian using classical Simulated Annealing.
    
    Parameters:
    -----------
    bqm : dimod.BinaryQuadraticModel
        Ising Hamiltonian to optimize
    num_reads : int
        Number of independent SA runs
    num_sweeps : int
        Number of sweeps per run
    beta_range : tuple
        Inverse temperature range (beta_min, beta_max) for annealing schedule
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    Tuple[Dict[str, int], float, Dict]
        Best spin configuration, best energy, and metadata dictionary
    """
    # Initialize SA sampler
    sampler = SimulatedAnnealingSampler()
    
    # Run optimization
    start_time = time.time()
    response = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=beta_range,
        seed=seed
    )
    elapsed_time = time.time() - start_time
    
    # Extract best solution
    best_sample = response.first.sample
    best_energy = response.first.energy
    
    # Compute statistics
    energies = [sample.energy for sample in response.data()]
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    
    metadata = {
        'num_reads': num_reads,
        'num_sweeps': num_sweeps,
        'beta_range': beta_range,
        'best_energy': best_energy,
        'mean_energy': mean_energy,
        'std_energy': std_energy,
        'min_energy': np.min(energies),
        'max_energy': np.max(energies),
        'elapsed_time': elapsed_time,
        'seed': seed
    }
    
    return best_sample, best_energy, metadata


def interpret_spin_configuration(spin_config: Dict[str, int],
                                 spin_names: list = None) -> Dict[str, str]:
    """
    Interpret spin configuration as control system states.
    
    Parameters:
    -----------
    spin_config : dict
        Spin configuration {spin_name: spin_value} where spin_value ∈ {-1, +1}
    spin_names : list, optional
        List of spin variable names
        
    Returns:
    --------
    dict
        Human-readable control states {'system': 'ON'/'OFF'}
    """
    if spin_names is None:
        from build_hamiltonian import SPIN_VARIABLES
        spin_names = SPIN_VARIABLES
    
    interpretation = {}
    for name in spin_names:
        if name in spin_config:
            state = 'ON' if spin_config[name] == 1 else 'OFF'
            interpretation[name] = state
    
    return interpretation


def run_sa_optimization(bqm: dimod.BinaryQuadraticModel,
                       num_reads: int = 1000,
                       verbose: bool = True) -> Tuple[Dict[str, int], float, Dict]:
    """
    Run SA optimization with default parameters and return results.
    
    Parameters:
    -----------
    bqm : dimod.BinaryQuadraticModel
        Ising Hamiltonian to optimize
    num_reads : int
        Number of independent SA runs
    verbose : bool
        Whether to print optimization results
        
    Returns:
    --------
    Tuple[Dict[str, int], float, Dict]
        Best spin configuration, best energy, and metadata
    """
    if verbose:
        print(f"Running Simulated Annealing optimization...")
        print(f"  Variables: {len(bqm.variables)}")
        print(f"  Interactions: {len(bqm.quadratic)}")
        print(f"  Number of reads: {num_reads}")
    
    best_sample, best_energy, metadata = optimize_with_sa(
        bqm, num_reads=num_reads
    )
    
    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Best energy: {best_energy:.4f}")
        print(f"  Mean energy: {metadata['mean_energy']:.4f} ± {metadata['std_energy']:.4f}")
        print(f"  Elapsed time: {metadata['elapsed_time']:.2f} seconds")
        print(f"\nBest configuration:")
        interpretation = interpret_spin_configuration(best_sample)
        for system, state in interpretation.items():
            print(f"  {system}: {state}")
    
    return best_sample, best_energy, metadata


if __name__ == '__main__':
    # Example usage
    from build_hamiltonian import build_hamiltonian_from_data
    
    print("Testing Simulated Annealing optimization...")
    
    # Build Hamiltonian
    bqm, J, h = build_hamiltonian_from_data(
        sensor_data={'temperature': 20.0, 'humidity': 60.0},
        co2_penalty=0.0
    )
    
    # Optimize
    best_sample, best_energy, metadata = run_sa_optimization(bqm, num_reads=500)
    
    print(f"\nMetadata: {metadata}")

