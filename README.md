# Multi-Objective Climatic Optimisation of Agricultural Greenhouse Systems Using the Quantum Ising Model

This repository contains the complete computational reproducibility package for the academic article:

**"Multi-Objective Climatic Optimisation of Agricultural Greenhouse Systems Using the Quantum Ising Model"**

## Project Information

- **Dataset**: `greenhouse_data.xlsx` (real dataset from EasyFlex Smart Greenhouse Automation System)
- **Source**: TÜBİTAK Project 7240482 – "Topraksız Tarıma Yönelik Makine Öğrenimi Tabanlı Akıllı Sistem Çözümleri"
- **Author**: Cumali Yaşar
- **Year**: 2025
- **Type**: Computational Reproducibility Package

## Project Structure

```
greenhouse-qim-optimisation/
│
├─ data/
│   └─ greenhouse_data.xlsx          # Greenhouse sensor dataset
│
├─ src/
│   ├─ load_data.py                   # Data loading and preprocessing
│   ├─ eda_correlations.py            # Exploratory data analysis
│   ├─ build_hamiltonian.py           # Ising Hamiltonian construction
│   ├─ optimise_sa.py                 # Classical simulated annealing
│   ├─ optimise_qanneal.py            # Quantum annealing (D-Wave)
│   ├─ co2_sensitivity.py             # CO₂ sensitivity analysis
│   ├─ co2_ablation.py                # CO₂ ablation experiments
│   └─ utils_plotting.py              # Plotting utilities
│
├─ results/                            # Output directory for results (JSON)
│   ├─ co2_sensitivity.json
│   └─ co2_ablation.json
│
├─ rapor/                              # Output directory for Excel reports
│   ├─ 00_ozet_rapor.xlsx              # Summary report
│   ├─ 01_eda_sonuclari.xlsx           # EDA results
│   ├─ 02_co2_sensitivity_sonuclari.xlsx  # CO₂ sensitivity results
│   ├─ 03_co2_ablation_sonuclari.xlsx  # CO₂ ablation results
│   └─ 04_hamiltonian_model.xlsx       # Hamiltonian model
│
├─ figures/                            # Output directory for figures
│   ├─ correlation_heatmap.png
│   ├─ light_vs_temperature.png
│   ├─ temperature_vs_humidity.png
│   ├─ interaction_matrix.png
│   ├─ external_field.png
│   ├─ co2_sensitivity.png
│   └─ co2_ablation.png
│
├─ 01_exploratory_analysis.py          # EDA script
├─ 02_build_qim_model.py               # QIM model construction script
├─ 03_experiments_co2_tests.py         # CO₂ experiments script
├─ README.md                           # This file
├─ requirements.txt                    # Python dependencies
└─ CITATION.cff                        # Citation information
```

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your dataset**:
   - Ensure `greenhouse_data.xlsx` is in the `data/` directory

4. **Optional: D-Wave Quantum Annealing Setup** (for quantum optimization):
   - Sign up for D-Wave Leap account: https://www.dwavesys.com/solutions-and-products/cloud-platform/
   - Set environment variable: `export DWAVE_API_TOKEN="your-token-here"`
   - Or pass token directly in code (see `optimise_qanneal.py`)

## Usage

### 1. Exploratory Data Analysis

Run the EDA script to analyze the dataset and generate correlation plots:

```bash
python 01_exploratory_analysis.py
```

This will:
- Load and preprocess the greenhouse data
- Compute Pearson correlations
- Generate scatter plots (Light vs Temperature, Temperature vs Humidity, etc.)
- Save figures to `figures/`
- Save Excel report to `rapor/01_eda_sonuclari.xlsx`

### 2. Build Quantum Ising Model

Construct and visualize the Ising Hamiltonian:

```bash
python 02_build_qim_model.py
```

This will:
- Build interaction matrix J
- Build external field vector h
- Visualize the Hamiltonian components
- Save figures to `figures/`
- Save Excel report to `rapor/04_hamiltonian_model.xlsx`

### 3. Run CO₂ Experiments

Execute sensitivity and ablation experiments:

```bash
python 03_experiments_co2_tests.py
```

This will:
- Run CO₂ sensitivity analysis (varying penalty weights λ = [0.0, 0.25, 0.5, 1.0])
- Run CO₂ ablation study (forced OFF)
- Save results to `results/` (JSON) and figures to `figures/`
- Save Excel reports to `rapor/`:
  - `02_co2_sensitivity_sonuclari.xlsx`
  - `03_co2_ablation_sonuclari.xlsx`
  - `00_ozet_rapor.xlsx` (summary report)

### Individual Module Usage

You can also import and use individual modules:

```python
from src.load_data import preprocess_greenhouse_data
from src.build_hamiltonian import build_hamiltonian_from_data
from src.optimise_sa import run_sa_optimization

# Load data
df, limits = preprocess_greenhouse_data('data/greenhouse_data.xlsx')

# Build Hamiltonian
bqm, J, h = build_hamiltonian_from_data(
    sensor_data={'temperature': 20.0, 'humidity': 60.0},
    co2_penalty=0.0
)

# Optimize
best_sample, best_energy, metadata = run_sa_optimization(bqm, num_reads=1000)
```

## Hardware Requirements

- **Recommended**: 128 GB RAM, Dell multi-CPU system, RTX 4070 GPU
- **Minimum**: 8 GB RAM, modern CPU
- Quantum annealing requires internet connection and D-Wave API access

## Data Source

The dataset `greenhouse_data.xlsx` contains real sensor data from the EasyFlex Smart Greenhouse Automation System, collected as part of TÜBİTAK Project 7240482.

**Physical Limits** (for outlier detection):
- Temperature: 5.0 - 50.0 °C
- Humidity: 0.0 - 100.0 %
- CO₂: 200.0 - 2000.0 ppm
- pH: 4.0 - 9.0
- EC: 0.0 - 5.0 mS/cm

## Methodology

### Ising Hamiltonian

The optimization problem is formulated as an Ising Hamiltonian:

```
H = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ
```

where:
- `sᵢ ∈ {-1, +1}` are spin variables representing control systems
- `Jᵢⱼ` is the interaction strength between spins i and j
- `hᵢ` is the external field bias for spin i

### Spin Variables

- `heater`: Heating system
- `fan`: Ventilation fan
- `misting`: Misting system
- `LED`: LED lighting
- `CO2`: CO₂ injection
- `irrigation`: Irrigation system
- `pH`: pH control
- `EC`: EC (electrical conductivity) control

### Optimization Methods

1. **Classical Simulated Annealing** (`neal`): Default method, no external dependencies
2. **Quantum Annealing** (`dwave-system`): Requires D-Wave API credentials

## Responsible AI Usage

This code is provided for academic and research purposes. When using quantum computing resources:
- Be mindful of API usage limits
- Follow D-Wave's terms of service
- Consider environmental impact of computational resources

## Citation

If you use this code or dataset, please cite:

```bibtex
@software{greenhouse_qim_optimisation,
  author = {Cumali Yaşar},
  title = {Multi-Objective Climatic Optimisation of Agricultural Greenhouse Systems Using the Quantum Ising Model},
  year = {2025},
  type = {Computational Reproducibility Package},
  note = {TÜBİTAK Project 7240482}
}
```

See `CITATION.cff` for machine-readable citation information.

## License

This code is provided for academic and research purposes. Please refer to the article for full details on methodology and results.

## Contact

For questions or issues, please refer to the corresponding author of the article.

## Acknowledgments

- TÜBİTAK Project 7240482 – "Topraksız Tarıma Yönelik Makine Öğrenimi Tabanlı Akıllı Sistem Çözümleri"
- EasyFlex Smart Greenhouse Automation System
- D-Wave Systems (for quantum annealing capabilities)

