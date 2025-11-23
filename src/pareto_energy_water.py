"""
Pareto Front Analysis: Energy-Water Trade-offs Module.

This module explores multi-objective trade-offs by varying weights in the
external field (h_i) of the Ising Hamiltonian. Analyzes the Pareto front
between energy consumption and water usage across different optimization scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dimod
from typing import Dict, List, Tuple
import os

from build_hamiltonian import (
    build_interaction_matrix,
    build_external_field,
    build_ising_hamiltonian,
    SPIN_VARIABLES
)
from optimise_sa import optimize_with_sa


# Spin variable mapping to objectives
# Note: SPIN_VARIABLES = ['heater', 'fan', 'misting', 'LED', 'CO2', 'irrigation', 'pH', 'EC']
# Mapping:
# - Energy-consuming actuators: heater (0), fan (1), LED (3), CO2 (4)
# - Water-consuming actuators: irrigation (5)
# - Yield-enhancing actuators: LED (3), CO2 (4)
SPIN_TO_ENERGY = [0, 1, 3, 4]  # heater, fan, LED, CO2 indices
SPIN_TO_WATER = [5]             # irrigation index
SPIN_TO_YIELD = [3, 4]          # LED, CO2 indices


def build_weighted_hamiltonian(scenario: Dict,
                              n_spins: int = 8,
                              base_interaction_strength: float = 1.0) -> dimod.BinaryQuadraticModel:
    """
    Build Ising Hamiltonian with weighted external field based on scenario priorities.
    
    The external field coefficients are adjusted to reflect:
    - Energy cost: higher weight penalizes energy-consuming actuators (heater, fan, LED, CO2)
    - Water cost: higher weight penalizes water-consuming actuators (irrigation)
    - Yield benefit: higher weight favors yield-enhancing actuators (LED, CO2)
    
    Parameters:
    -----------
    scenario : dict
        Scenario dictionary with keys: 'name', 'w_energy', 'w_water', 'w_yield'
    n_spins : int
        Number of spin variables (default: 8)
    base_interaction_strength : float
        Base strength for interaction matrix
        
    Returns:
    --------
    dimod.BinaryQuadraticModel
        Weighted Ising Hamiltonian
    """
    # Build interaction matrix (unchanged across scenarios)
    J = build_interaction_matrix(n_spins, base_interaction_strength)
    
    # Build base external field
    h_base = build_external_field(n_spins, field_strength=1.0)
    
    # Apply scenario-specific weights to external field
    h_weighted = h_base.copy()
    
    w_energy = scenario.get('w_energy', 1.0)
    w_water = scenario.get('w_water', 1.0)
    w_yield = scenario.get('w_yield', 1.0)
    
    # Energy cost: positive bias (prefer OFF) for energy-consuming actuators
    # In Ising model, positive h_i favors s_i = -1 (OFF)
    for idx in SPIN_TO_ENERGY:
        h_weighted[idx] += w_energy * 0.3  # Penalty for energy consumption
    
    # Water cost: positive bias (prefer OFF) for water-consuming actuators
    for idx in SPIN_TO_WATER:
        h_weighted[idx] += w_water * 0.4  # Penalty for water consumption
    
    # Yield benefit: negative bias (prefer ON) for yield-enhancing actuators
    # In Ising model, negative h_i favors s_i = +1 (ON)
    for idx in SPIN_TO_YIELD:
        h_weighted[idx] -= w_yield * 0.2  # Reward for yield enhancement
    
    # Build BQM
    bqm = build_ising_hamiltonian(J, h_weighted, SPIN_VARIABLES)
    
    return bqm


def count_water_activations(spin_config: Dict[str, int]) -> int:
    """
    Count number of water-related actuators that are ON.
    
    Parameters:
    -----------
    spin_config : dict
        Spin configuration {spin_name: spin_value} where spin_value ∈ {-1, +1}
        
    Returns:
    --------
    int
        Number of water-related actuators in ON state (spin = +1)
    """
    water_spins = ['irrigation']  # Only irrigation uses water directly
    count = 0
    for spin_name in water_spins:
        if spin_name in spin_config and spin_config[spin_name] == 1:
            count += 1
    return count


def get_co2_state(spin_config: Dict[str, int]) -> str:
    """
    Get CO2 actuator state.
    
    Parameters:
    -----------
    spin_config : dict
        Spin configuration
        
    Returns:
    --------
    str
        'ON' if CO2 is active (spin = +1), 'OFF' otherwise
    """
    if 'CO2' in spin_config:
        return 'ON' if spin_config['CO2'] == 1 else 'OFF'
    return 'UNKNOWN'


def run_scenario_optimization(scenario: Dict,
                             num_reads: int = 1000) -> Dict:
    """
    Run optimization for a single scenario and collect results.
    
    Parameters:
    -----------
    scenario : dict
        Scenario dictionary with weights
    num_reads : int
        Number of SA reads
        
    Returns:
    --------
    dict
        Results dictionary with keys: 'scenario', 'energy', 'water_activation_count', 'co2_state'
    """
    # Build weighted Hamiltonian
    bqm = build_weighted_hamiltonian(scenario)
    
    # Run optimization
    best_sample, best_energy, metadata = optimize_with_sa(
        bqm, num_reads=num_reads, seed=42
    )
    
    # Extract metrics
    water_count = count_water_activations(best_sample)
    co2_state = get_co2_state(best_sample)
    
    return {
        'scenario': scenario['name'],
        'energy': best_energy,
        'water_activation_count': water_count,
        'co2_state': co2_state
    }


def compute_pareto_front(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pareto front from results (non-dominated solutions).
    
    A solution is Pareto-optimal if no other solution is better in both objectives.
    Here we minimize both energy and water_activation_count.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe with 'energy' and 'water_activation_count' columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe containing only Pareto-optimal solutions
    """
    pareto_indices = []
    
    for i, row_i in results_df.iterrows():
        is_pareto = True
        for j, row_j in results_df.iterrows():
            if i == j:
                continue
            # Check if j dominates i (both objectives better)
            if (row_j['energy'] <= row_i['energy'] and 
                row_j['water_activation_count'] <= row_i['water_activation_count'] and
                (row_j['energy'] < row_i['energy'] or 
                 row_j['water_activation_count'] < row_i['water_activation_count'])):
                is_pareto = False
                break
        if is_pareto:
            pareto_indices.append(i)
    
    return results_df.loc[pareto_indices].copy()


def plot_pareto_front(results_df: pd.DataFrame,
                      save_path: str = 'figures/pareto_energy_water.png'):
    """
    Plot Pareto front: energy vs water activation count.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        Results dataframe
    save_path : str
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color map for scenarios
    scenarios = results_df['scenario'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(scenarios)))
    color_map = {scenario: colors[i] for i, scenario in enumerate(scenarios)}
    
    # Plot all points
    for _, row in results_df.iterrows():
        ax.scatter(row['energy'], row['water_activation_count'],
                  c=[color_map[row['scenario']]], s=200, alpha=0.7,
                  edgecolors='black', linewidth=1.5, label=row['scenario'])
    
    # Add labels
    for _, row in results_df.iterrows():
        ax.annotate(row['scenario'], 
                   (row['energy'], row['water_activation_count']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Hamiltonian Energy (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Water Activation Count', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Front: Energy vs Water Trade-offs', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Pareto front plot to {save_path}")
    plt.close()


def main():
    """
    Main function: run Pareto front analysis across multiple scenarios.
    """
    print("=" * 80)
    print("PARETO FRONT ANALYSIS: ENERGY-WATER TRADE-OFFS")
    print("=" * 80)
    
    # Define optimization scenarios
    scenarios = [
        {"name": "energy_saving",  "w_energy": 1.5, "w_water": 1.0, "w_yield": 0.8},
        {"name": "yield_priority", "w_energy": 1.0, "w_water": 1.0, "w_yield": 1.5},
        {"name": "water_saving",   "w_energy": 1.0, "w_water": 1.5, "w_yield": 1.0},
        {"name": "balanced",       "w_energy": 1.2, "w_water": 1.2, "w_yield": 1.2},
    ]
    
    print(f"\nAnalyzing {len(scenarios)} scenarios...")
    print("-" * 80)
    
    # Run optimization for each scenario
    results = []
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Scenario: {scenario['name']}")
        print(f"  Weights: energy={scenario['w_energy']}, "
              f"water={scenario['w_water']}, yield={scenario['w_yield']}")
        
        result = run_scenario_optimization(scenario, num_reads=1000)
        results.append(result)
        
        print(f"  Result: energy={result['energy']:.4f}, "
              f"water_count={result['water_activation_count']}, "
              f"CO2={result['co2_state']}")
    
    # Build results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Plot Pareto front
    print("\n" + "=" * 80)
    print("Generating Pareto front plot...")
    plot_pareto_front(results_df)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: figures/pareto_energy_water.png")
    print()


if __name__ == '__main__':
    main()

