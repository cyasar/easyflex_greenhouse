"""
Quantum Annealing optimization module using D-Wave Ocean.

This module implements quantum optimization using D-Wave's quantum annealers
via DWaveSampler and EmbeddingComposite for the greenhouse Ising Hamiltonian.

NOTE: Requires D-Wave API credentials (token and solver endpoint).
"""

import numpy as np
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite
from typing import Dict, Tuple, Optional
import time
import os


def optimize_with_quantum_annealing(bqm: dimod.BinaryQuadraticModel,
                                    num_reads: int = 1000,
                                    chain_strength: Optional[float] = None,
                                    token: Optional[str] = None,
                                    endpoint: Optional[str] = None,
                                    solver: Optional[str] = None) -> Tuple[Dict[str, int], float, Dict]:
    """
    Optimize Ising Hamiltonian using D-Wave quantum annealer.
    
    Parameters:
    -----------
    bqm : dimod.BinaryQuadraticModel
        Ising Hamiltonian to optimize
    num_reads : int
        Number of reads from the quantum annealer
    chain_strength : float, optional
        Chain strength for embedding. If None, auto-calculated.
    token : str, optional
        D-Wave API token. If None, reads from DWAVE_API_TOKEN env var.
    endpoint : str, optional
        D-Wave API endpoint. If None, uses default.
    solver : str, optional
        Solver name. If None, uses default available solver.
        
    Returns:
    --------
    Tuple[Dict[str, int], float, Dict]
        Best spin configuration, best energy, and metadata dictionary
    """
    # Get API token
    if token is None:
        token = os.getenv('DWAVE_API_TOKEN')
        if token is None:
            raise ValueError(
                "D-Wave API token required. Set DWAVE_API_TOKEN environment variable "
                "or pass token parameter."
            )
    
    # Initialize D-Wave sampler
    try:
        dwave_sampler = DWaveSampler(
            token=token,
            endpoint=endpoint,
            solver=solver
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to D-Wave: {e}\n"
            "Please ensure you have valid D-Wave API credentials."
        )
    
    # Use EmbeddingComposite to handle minor embedding
    sampler = EmbeddingComposite(dwave_sampler)
    
    # Auto-calculate chain strength if not provided
    if chain_strength is None:
        # Use a heuristic: 1.5 * max absolute bias
        max_bias = max(
            max(abs(b) for b in bqm.linear.values()) if bqm.linear else [0],
            max(abs(b) for b in bqm.quadratic.values()) if bqm.quadratic else [0]
        )
        chain_strength = 1.5 * max_bias if max_bias > 0 else 1.0
    
    # Run optimization
    start_time = time.time()
    try:
        response = sampler.sample(
            bqm,
            num_reads=num_reads,
            chain_strength=chain_strength,
            return_embedding=True
        )
    except Exception as e:
        raise RuntimeError(f"Quantum annealing failed: {e}")
    
    elapsed_time = time.time() - start_time
    
    # Extract best solution
    best_sample = response.first.sample
    best_energy = response.first.energy
    
    # Compute statistics
    energies = [sample.energy for sample in response.data()]
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)
    
    # Extract embedding info if available
    embedding_info = {}
    if hasattr(response, 'info') and 'embedding' in response.info:
        embedding_info = {
            'chain_lengths': [len(chain) for chain in response.info['embedding'].values()],
            'max_chain_length': max(len(chain) for chain in response.info['embedding'].values()) if response.info['embedding'] else 0
        }
    
    metadata = {
        'num_reads': num_reads,
        'chain_strength': chain_strength,
        'best_energy': best_energy,
        'mean_energy': mean_energy,
        'std_energy': std_energy,
        'min_energy': np.min(energies),
        'max_energy': np.max(energies),
        'elapsed_time': elapsed_time,
        'solver': str(dwave_sampler.solver),
        'embedding_info': embedding_info
    }
    
    return best_sample, best_energy, metadata


def run_quantum_optimization(bqm: dimod.BinaryQuadraticModel,
                            num_reads: int = 1000,
                            verbose: bool = True,
                            token: Optional[str] = None) -> Tuple[Dict[str, int], float, Dict]:
    """
    Run quantum annealing optimization with default parameters.
    
    Parameters:
    -----------
    bqm : dimod.BinaryQuadraticModel
        Ising Hamiltonian to optimize
    num_reads : int
        Number of reads from the quantum annealer
    verbose : bool
        Whether to print optimization results
    token : str, optional
        D-Wave API token. If None, reads from environment.
        
    Returns:
    --------
    Tuple[Dict[str, int], float, Dict]
        Best spin configuration, best energy, and metadata
    """
    if verbose:
        print(f"Running Quantum Annealing optimization...")
        print(f"  Variables: {len(bqm.variables)}")
        print(f"  Interactions: {len(bqm.quadratic)}")
        print(f"  Number of reads: {num_reads}")
        print(f"  NOTE: This requires D-Wave API credentials")
    
    best_sample, best_energy, metadata = optimize_with_quantum_annealing(
        bqm, num_reads=num_reads, token=token
    )
    
    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Best energy: {best_energy:.4f}")
        print(f"  Mean energy: {metadata['mean_energy']:.4f} ± {metadata['std_energy']:.4f}")
        print(f"  Elapsed time: {metadata['elapsed_time']:.2f} seconds")
        print(f"  Solver: {metadata['solver']}")
        if metadata['embedding_info']:
            print(f"  Max chain length: {metadata['embedding_info'].get('max_chain_length', 'N/A')}")
        print(f"\nBest configuration:")
        from optimise_sa import interpret_spin_configuration
        interpretation = interpret_spin_configuration(best_sample)
        for system, state in interpretation.items():
            print(f"  {system}: {state}")
    
    return best_sample, best_energy, metadata


if __name__ == '__main__':
    # Example usage (requires D-Wave credentials)
    from build_hamiltonian import build_hamiltonian_from_data
    
    print("Testing Quantum Annealing optimization...")
    print("NOTE: This requires D-Wave API credentials (DWAVE_API_TOKEN)")
    
    # Build Hamiltonian
    bqm, J, h = build_hamiltonian_from_data(
        sensor_data={'temperature': 20.0, 'humidity': 60.0},
        co2_penalty=0.0
    )
    
    # Check if credentials are available
    token = os.getenv('DWAVE_API_TOKEN')
    if token:
        try:
            # Optimize
            best_sample, best_energy, metadata = run_quantum_optimization(
                bqm, num_reads=100, token=token
            )
            print(f"\nMetadata: {metadata}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Skipping quantum optimization - DWAVE_API_TOKEN not set")

