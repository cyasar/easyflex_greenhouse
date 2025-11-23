"""
CO₂ Sensitivity and Ablation Experiments Script

This script runs CO₂ sensitivity and ablation experiments:
- CO₂ sensitivity analysis with varying penalty weights
- CO₂ ablation study (forced OFF)
- Results visualization and comparison
"""

import sys
import os
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import json

from build_hamiltonian import build_hamiltonian_from_data
from optimise_sa import run_sa_optimization, interpret_spin_configuration
from co2_sensitivity import (
    run_co2_sensitivity_analysis,
    save_sensitivity_results,
    plot_sensitivity_curve
)
from co2_ablation import (
    run_co2_ablation_study,
    save_ablation_results,
    plot_ablation_comparison
)
from utils_plotting import setup_plot_style
from report_excel import save_sensitivity_to_excel, save_ablation_to_excel, create_summary_report

# Setup plotting style
setup_plot_style()


def main():
    """Main function for CO₂ experiments."""
    
    print("=" * 60)
    print("CO₂ Sensitivity and Ablation Experiments")
    print("=" * 60)
    
    # Sensor data for experiments
    sensor_data = {
        'temperature': 20.0,
        'humidity': 60.0,
        'co2': 400.0
    }
    
    # 1. CO₂ Sensitivity Analysis
    print("\n" + "=" * 60)
    print("1. CO₂ Sensitivity Analysis")
    print("=" * 60)
    print("Testing different CO₂ penalty weights (lambda)...")
    
    lambda_values = [0.0, 0.25, 0.5, 1.0]
    
    sensitivity_results = run_co2_sensitivity_analysis(
        sensor_data=sensor_data,
        lambda_values=lambda_values,
        num_reads=1000,
        use_quantum=False  # Set to True if D-Wave credentials available
    )
    
    # Save results
    save_sensitivity_results(sensitivity_results, 'results/co2_sensitivity.json')
    
    # Save to Excel
    save_sensitivity_to_excel(sensitivity_results, 'rapor/02_co2_sensitivity_sonuclari.xlsx')
    
    # Plot sensitivity curve
    plot_sensitivity_curve(sensitivity_results, 'figures/co2_sensitivity.png')
    
    # Print summary
    print("\nSensitivity Analysis Summary:")
    print("-" * 40)
    for i, lam in enumerate(lambda_values):
        energy = sensitivity_results['energies'][i]
        config = sensitivity_results['configurations'][i]
        co2_state = 'ON' if config.get('CO2', -1) == 1 else 'OFF'
        print(f"Lambda = {lam:4.2f}: Energy = {energy:8.4f}, CO₂ = {co2_state}")
    
    # 2. CO₂ Ablation Study
    print("\n" + "=" * 60)
    print("2. CO₂ Ablation Study")
    print("=" * 60)
    print("Comparing normal optimization vs CO₂ forced OFF...")
    
    ablation_results = run_co2_ablation_study(
        sensor_data=sensor_data,
        num_reads=1000,
        co2_penalty=0.0
    )
    
    # Save results
    save_ablation_results(ablation_results, 'results/co2_ablation.json')
    
    # Save to Excel
    save_ablation_to_excel(ablation_results, 'rapor/03_co2_ablation_sonuclari.xlsx')
    
    # Plot ablation comparison
    plot_ablation_comparison(ablation_results, 'figures/co2_ablation.png')
    
    # Print summary
    print("\nAblation Study Summary:")
    print("-" * 40)
    print(f"Normal case energy:    {ablation_results['normal']['energy']:8.4f}")
    print(f"Ablation case energy:  {ablation_results['ablation']['energy']:8.4f}")
    print(f"Energy difference:     {ablation_results['comparison']['energy_difference']:8.4f}")
    print(f"Energy ratio:          {ablation_results['comparison']['energy_ratio']:8.4f}")
    
    # 3. Detailed Configuration Comparison
    print("\n" + "=" * 60)
    print("3. Configuration Comparison")
    print("=" * 60)
    
    print("\nNormal case configuration:")
    normal_config = interpret_spin_configuration(
        ablation_results['normal']['configuration']
    )
    for system, state in normal_config.items():
        print(f"  {system:12s}: {state}")
    
    print("\nAblation case configuration (CO₂ forced OFF):")
    ablation_config = interpret_spin_configuration(
        ablation_results['ablation']['configuration']
    )
    for system, state in ablation_config.items():
        print(f"  {system:12s}: {state}")
    
    # 4. Create Summary Report
    print("\n" + "=" * 60)
    print("4. Creating Summary Report")
    print("=" * 60)
    create_summary_report(
        eda_file='rapor/01_eda_sonuclari.xlsx',
        sensitivity_file='rapor/02_co2_sensitivity_sonuclari.xlsx',
        ablation_file='rapor/03_co2_ablation_sonuclari.xlsx',
        hamiltonian_file='rapor/04_hamiltonian_model.xlsx',
        filepath='rapor/00_ozet_rapor.xlsx'
    )
    
    # 5. Summary
    print("\n" + "=" * 60)
    print("Experiments Complete!")
    print("=" * 60)
    print("Results saved to: results/")
    print("Figures saved to: figures/")
    print("Excel reports saved to: rapor/")


if __name__ == '__main__':
    main()

