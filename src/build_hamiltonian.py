"""
Ising Hamiltonian construction module for greenhouse optimization.

This module builds the quantum Ising model (QIM) Hamiltonian using dimod
for multi-objective optimization of greenhouse control systems.
"""

import numpy as np
import dimod
from typing import Dict, List, Tuple, Optional


# Spin variable names (control systems)
SPIN_VARIABLES = [
    'heater',      # Spin 0: Heating system
    'fan',         # Spin 1: Ventilation fan
    'misting',     # Spin 2: Misting system
    'LED',         # Spin 3: LED lighting
    'CO2',         # Spin 4: CO2 injection
    'irrigation',  # Spin 5: Irrigation system
    'pH',          # Spin 6: pH control
    'EC'           # Spin 7: EC (electrical conductivity) control
]


def build_interaction_matrix(n_spins: int = 8, 
                            interaction_strength: float = 1.0,
                            custom_J: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build interaction matrix J for Ising Hamiltonian.
    
    The interaction matrix defines pairwise couplings between spins.
    In greenhouse context, this represents interactions between control systems.
    
    Parameters:
    -----------
    n_spins : int
        Number of spin variables (default: 8)
    interaction_strength : float
        Base strength of interactions
    custom_J : np.ndarray, optional
        Custom interaction matrix. If provided, overrides default.
        
    Returns:
    --------
    np.ndarray
        Interaction matrix J of shape (n_spins, n_spins)
    """
    if custom_J is not None:
        if custom_J.shape != (n_spins, n_spins):
            raise ValueError(f"Custom J must be {n_spins}x{n_spins}")
        return custom_J
    
    # Default: symmetric interaction matrix
    # Can be customized based on greenhouse physics
    J = np.zeros((n_spins, n_spins))
    
    # Example interactions (can be tuned based on data/domain knowledge):
    # - Heater and Fan are anti-correlated (heating vs cooling)
    # - LED and CO2 may be correlated (both for plant growth)
    # - Irrigation and pH/EC are correlated (nutrient management)
    
    # Anti-correlation: heater vs fan
    J[0, 1] = -interaction_strength * 0.5
    J[1, 0] = -interaction_strength * 0.5
    
    # Correlation: LED and CO2
    J[3, 4] = interaction_strength * 0.3
    J[4, 3] = interaction_strength * 0.3
    
    # Correlation: irrigation, pH, EC
    J[5, 6] = interaction_strength * 0.4
    J[6, 5] = interaction_strength * 0.4
    J[5, 7] = interaction_strength * 0.4
    J[7, 5] = interaction_strength * 0.4
    J[6, 7] = interaction_strength * 0.5
    J[7, 6] = interaction_strength * 0.5
    
    return J


def build_external_field(n_spins: int = 8,
                        field_strength: float = 1.0,
                        custom_h: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Build external field vector h for Ising Hamiltonian.
    
    The external field represents bias towards certain spin states
    based on environmental conditions or optimization objectives.
    
    Parameters:
    -----------
    n_spins : int
        Number of spin variables (default: 8)
    field_strength : float
        Base strength of external field
    custom_h : np.ndarray, optional
        Custom external field vector. If provided, overrides default.
        
    Returns:
    --------
    np.ndarray
        External field vector h of shape (n_spins,)
    """
    if custom_h is not None:
        if len(custom_h) != n_spins:
            raise ValueError(f"Custom h must have length {n_spins}")
        return custom_h
    
    # Default: uniform external field
    # Can be customized based on sensor readings or optimization goals
    h = np.ones(n_spins) * field_strength * 0.1
    
    return h


def build_ising_hamiltonian(J: np.ndarray, 
                           h: np.ndarray,
                           spin_names: List[str] = None) -> dimod.BinaryQuadraticModel:
    """
    Build Ising Hamiltonian as a Binary Quadratic Model (BQM) using dimod.
    
    The Hamiltonian is: H = -sum_ij J_ij * s_i * s_j - sum_i h_i * s_i
    where s_i ∈ {-1, +1} are spin variables.
    
    Parameters:
    -----------
    J : np.ndarray
        Interaction matrix of shape (n_spins, n_spins)
    h : np.ndarray
        External field vector of shape (n_spins,)
    spin_names : list of str, optional
        Names for spin variables. If None, uses default SPIN_VARIABLES.
        
    Returns:
    --------
    dimod.BinaryQuadraticModel
        Ising Hamiltonian as a BQM in SPIN representation
    """
    n_spins = len(h)
    
    if J.shape != (n_spins, n_spins):
        raise ValueError(f"J must be {n_spins}x{n_spins}")
    
    if spin_names is None:
        spin_names = SPIN_VARIABLES[:n_spins]
    
    if len(spin_names) != n_spins:
        raise ValueError(f"spin_names must have length {n_spins}")
    
    # Initialize BQM in SPIN representation
    bqm = dimod.BinaryQuadraticModel.empty(dimod.SPIN)
    
    # Add linear terms (external field): -h_i * s_i
    for i, name in enumerate(spin_names):
        bqm.add_variable(name, -h[i])
    
    # Add quadratic terms (interactions): -J_ij * s_i * s_j
    for i in range(n_spins):
        for j in range(i + 1, n_spins):
            if abs(J[i, j]) > 1e-10:  # Only add non-zero interactions
                bqm.add_interaction(spin_names[i], spin_names[j], -J[i, j])
    
    return bqm


def build_hamiltonian_from_data(sensor_data: Dict[str, float] = None,
                                n_spins: int = 8,
                                interaction_strength: float = 1.0,
                                field_strength: float = 1.0,
                                co2_penalty: float = 0.0) -> Tuple[dimod.BinaryQuadraticModel, np.ndarray, np.ndarray]:
    """
    Build complete Ising Hamiltonian from sensor data and parameters.
    
    Parameters:
    -----------
    sensor_data : dict, optional
        Dictionary of sensor readings (temperature, humidity, co2, etc.)
        Used to adjust external field based on environmental conditions
    n_spins : int
        Number of spin variables
    interaction_strength : float
        Base interaction strength
    field_strength : float
        Base external field strength
    co2_penalty : float
        Additional penalty weight for CO2 usage (lambda in sensitivity tests)
        
    Returns:
    --------
    Tuple[dimod.BinaryQuadraticModel, np.ndarray, np.ndarray]
        BQM Hamiltonian, interaction matrix J, external field h
    """
    # Build interaction matrix
    J = build_interaction_matrix(n_spins, interaction_strength)
    
    # Build external field
    h = build_external_field(n_spins, field_strength)
    
    # Adjust external field based on sensor data if provided
    if sensor_data is not None:
        # Example: if temperature is low, bias heater towards +1 (on)
        if 'temperature' in sensor_data:
            temp = sensor_data['temperature']
            # Normalize temperature (assuming 15-30°C range)
            temp_norm = (temp - 15) / 15  # Maps 15°C -> 0, 30°C -> 1
            h[0] += -0.5 * temp_norm  # Heater bias (negative = prefer +1 when cold)
            h[1] += 0.5 * temp_norm   # Fan bias (positive = prefer +1 when hot)
        
        # CO2 penalty adjustment
        if co2_penalty > 0:
            # Add penalty to CO2 spin (index 4) to discourage usage
            h[4] += co2_penalty
    
    # Build BQM
    bqm = build_ising_hamiltonian(J, h)
    
    return bqm, J, h


def compute_hamiltonian_energy(bqm: dimod.BinaryQuadraticModel,
                              spin_config: Dict[str, int]) -> float:
    """
    Compute Hamiltonian energy for a given spin configuration.
    
    Parameters:
    -----------
    bqm : dimod.BinaryQuadraticModel
        Ising Hamiltonian
    spin_config : dict
        Spin configuration {spin_name: spin_value} where spin_value ∈ {-1, +1}
        
    Returns:
    --------
    float
        Hamiltonian energy
    """
    return bqm.energy(spin_config)


if __name__ == '__main__':
    # Example usage
    print("Building Ising Hamiltonian for greenhouse optimization...")
    
    bqm, J, h = build_hamiltonian_from_data(
        sensor_data={'temperature': 20.0, 'humidity': 60.0},
        co2_penalty=0.0
    )
    
    print(f"\nNumber of variables: {len(bqm.variables)}")
    print(f"Number of interactions: {len(bqm.quadratic)}")
    print(f"\nInteraction matrix J:\n{J}")
    print(f"\nExternal field h:\n{h}")
    print(f"\nBQM linear biases: {dict(bqm.linear)}")
    print(f"BQM quadratic biases (sample): {dict(list(bqm.quadratic.items())[:5])}")

