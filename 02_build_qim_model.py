"""
Quantum Ising Model (QIM) Construction Script

This script builds and visualizes the Ising Hamiltonian for greenhouse optimization,
including:
- Interaction matrix construction
- External field definition
- Hamiltonian visualization
- Model explanation
"""

import sys
import os
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt

from build_hamiltonian import (
    build_hamiltonian_from_data,
    build_interaction_matrix,
    build_external_field,
    SPIN_VARIABLES
)
from utils_plotting import (
    setup_plot_style,
    plot_interaction_matrix,
    plot_external_field,
    plot_spin_configuration
)
from report_excel import save_hamiltonian_to_excel

# Setup plotting style
setup_plot_style()


def main():
    """Main function for QIM model construction."""
    
    print("=" * 60)
    print("Quantum Ising Model (QIM) Construction")
    print("=" * 60)
    
    # 1. Build Interaction Matrix
    print("\n1. Building Interaction Matrix J...")
    J = build_interaction_matrix(n_spins=8, interaction_strength=1.0)
    
    print(f"Interaction matrix shape: {J.shape}")
    print(f"Spin variables: {SPIN_VARIABLES}")
    print("\nInteraction Matrix J:")
    print(J)
    
    # Visualize interaction matrix
    plot_interaction_matrix(J, spin_names=SPIN_VARIABLES,
                           save_path='figures/interaction_matrix.png',
                           title='Ising Model Interaction Matrix')
    
    # 2. Build External Field
    print("\n" + "=" * 60)
    print("2. Building External Field h...")
    print("=" * 60)
    
    h = build_external_field(n_spins=8, field_strength=1.0)
    
    print(f"External field shape: {h.shape}")
    print("\nExternal Field h:")
    print(h)
    
    # Visualize external field
    plot_external_field(h, spin_names=SPIN_VARIABLES,
                       save_path='figures/external_field.png',
                       title='Ising Model External Field')
    
    # 3. Build Complete Hamiltonian
    print("\n" + "=" * 60)
    print("3. Building Complete Ising Hamiltonian...")
    print("=" * 60)
    
    sensor_data = {
        'temperature': 20.0,
        'humidity': 60.0,
        'co2': 400.0
    }
    
    bqm, J_full, h_full = build_hamiltonian_from_data(
        sensor_data=sensor_data,
        n_spins=8,
        interaction_strength=1.0,
        field_strength=1.0,
        co2_penalty=0.0
    )
    
    print(f"BQM variables: {len(bqm.variables)}")
    print(f"BQM interactions: {len(bqm.quadratic)}")
    print("\nLinear biases (external field):")
    for var, bias in bqm.linear.items():
        print(f"  {var}: {bias:.4f}")
    
    print("\nQuadratic biases (interactions) - sample:")
    for i, (pair, bias) in enumerate(bqm.quadratic.items()):
        if i < 5:  # Show first 5
            print(f"  {pair}: {bias:.4f}")
    
    # 4. Model Explanation
    print("\n" + "=" * 60)
    print("4. Model Explanation")
    print("=" * 60)
    print("""
    The Ising Hamiltonian is defined as:
    
    H = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ
    
    where:
    - sᵢ ∈ {-1, +1} are spin variables representing control systems
    - Jᵢⱼ is the interaction strength between spins i and j
    - hᵢ is the external field bias for spin i
    
    Spin Variables:
    - heater: Heating system (s₀)
    - fan: Ventilation fan (s₁)
    - misting: Misting system (s₂)
    - LED: LED lighting (s₃)
    - CO2: CO₂ injection (s₄)
    - irrigation: Irrigation system (s₅)
    - pH: pH control (s₆)
    - EC: EC control (s₇)
    
    Optimization Goal:
    Find the spin configuration that minimizes H, representing
    the optimal control system state for the greenhouse.
    """)
    
    # 5. Save to Excel
    print("\n" + "=" * 60)
    print("5. Saving Model to Excel")
    print("=" * 60)
    save_hamiltonian_to_excel(J_full, h_full, SPIN_VARIABLES, 
                             'rapor/04_hamiltonian_model.xlsx')
    
    print("\n" + "=" * 60)
    print("QIM Model Construction Complete!")
    print("=" * 60)
    print("Figures saved to: figures/")
    print("Excel report saved to: rapor/04_hamiltonian_model.xlsx")


if __name__ == '__main__':
    main()

