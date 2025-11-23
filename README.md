# Multi-Objective Climatic Optimisation of Agricultural Greenhouse Systems Using the Quantum Ising Model

This repository contains the complete computational reproducibility package for the academic article:

**"Multi-Objective Climatic Optimisation of Agricultural Greenhouse Systems Using the Quantum Ising Model"**

## Project Information

- **Dataset**: `greenhouse_data.xlsx` (real dataset from EasyFlex Smart Greenhouse Automation System)
- **Source**: TÜBİTAK Project 7240482 – "Topraksız Tarıma Yönelik Makine Öğrenimi Tabanlı Akıllı Sistem Çözümleri"
- **Author**: Cumali Yaşar
- **Year**: 2025
- **Type**: Computational Reproducibility Package

## Repository Structure

```
QIM-Greenhouse-Optimization/
│
├─ data/                              # Dataset directory
│   └─ greenhouse_data.xlsx          # Greenhouse sensor dataset
│
├─ preprocessing/                     # Data preprocessing modules
│   ├─ load_data.py                  # Data loading and cleaning
│   ├─ eda_correlations.py           # Exploratory data analysis
│   └─ daily_profiles.py             # Diurnal profile analysis
│
├─ analysis/                         # Main analysis modules
│   ├─ build_hamiltonian.py          # Ising Hamiltonian construction
│   ├─ optimise_sa.py                # Classical simulated annealing
│   ├─ co2_sensitivity.py            # CO₂ sensitivity analysis
│   ├─ co2_ablation.py               # CO₂ ablation experiments
│   ├─ pareto_energy_water.py        # Pareto front analysis
│   ├─ solution_stability.py         # Solution stability analysis
│   ├─ noise_robustness.py           # Noise robustness analysis
│   ├─ utils_plotting.py             # Plotting utilities
│   ├─ report_excel.py               # Excel reporting utilities
│   │
│   └─ scripts/                      # Main execution scripts
│       ├─ 00_run_all.py             # Run all analyses
│       ├─ 01_exploratory_analysis.py
│       ├─ 02_build_qim_model.py
│       └─ 03_experiments_co2_tests.py
│
├─ dwave_annealing/                  # D-Wave quantum annealing
│   └─ optimise_qanneal.py           # Quantum annealing implementation
│
├─ qiskit_simulations/               # Qiskit quantum simulations (future)
│   └─ README.md                     # Placeholder for Qiskit code
│
├─ results/                          # Output directory
│   ├─ figures/                      # Generated plots
│   ├─ reports/                      # Excel reports
│   └─ json/                         # JSON result files
│
├─ README.md                         # This file
├─ requirements.txt                  # Python dependencies
├─ .gitignore                        # Git ignore rules
└─ CITATION.cff                      # Citation information
```

## Environment Requirements

### Python Version
- **Python 3.8+** (tested with Python 3.9, 3.10, 3.11)

### Operating System
- **Linux** (recommended)
- **Windows 10/11**
- **macOS**

### Hardware Recommendations
- **Minimum**: 8 GB RAM, modern CPU
- **Recommended**: 128 GB RAM, multi-core CPU, RTX 4070 GPU (for large-scale simulations)

## Package Versions

All required packages are specified in `requirements.txt`. Key dependencies:

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
dimod>=0.12.0
dwave-neal>=0.5.0
dwave-system>=1.20.0
scikit-learn>=1.3.0
openpyxl>=3.1.0
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/USERNAME/QIM-Greenhouse-Optimization.git
cd QIM-Greenhouse-Optimization
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare Dataset

Place your `greenhouse_data.xlsx` file in the `data/` directory:

```bash
cp /path/to/greenhouse_data.xlsx data/
```

### 5. Optional: D-Wave Quantum Annealing Setup

For quantum annealing experiments:

1. Sign up for D-Wave Leap account:
   https://www.dwavesys.com/solutions-and-products/cloud-platform/

2. Get your API token from the D-Wave Leap dashboard

3. Set environment variable:
   ```bash
   # Linux/macOS:
   export DWAVE_API_TOKEN="your-token-here"
   
   # Windows:
   set DWAVE_API_TOKEN=your-token-here
   
   # Or add to .bashrc/.zshrc for persistence:
   echo 'export DWAVE_API_TOKEN="your-token-here"' >> ~/.bashrc
   ```

## How to Reproduce Optimization Results

### Quick Start: Run All Analyses

```bash
cd analysis/scripts
python 00_run_all.py
```

This will execute all analyses in sequence:
1. Exploratory Data Analysis (EDA)
2. Quantum Ising Model Construction
3. CO₂ Sensitivity and Ablation Experiments

### Step-by-Step Execution

#### Step 1: Data Preprocessing and EDA

```bash
cd analysis/scripts
python 01_exploratory_analysis.py
```

**Outputs:**
- `results/figures/` - Correlation plots, scatter plots
- `results/reports/01_eda_sonuclari.xlsx` - EDA summary report

#### Step 2: Build Quantum Ising Model

```bash
python 02_build_qim_model.py
```

**Outputs:**
- `results/figures/interaction_matrix.png`
- `results/figures/external_field.png`
- `results/reports/04_hamiltonian_model.xlsx`

#### Step 3: CO₂ Experiments

```bash
python 03_experiments_co2_tests.py
```

**Outputs:**
- `results/figures/co2_sensitivity.png`
- `results/figures/co2_ablation.png`
- `results/json/co2_sensitivity.json`
- `results/json/co2_ablation.json`
- `results/reports/02_co2_sensitivity_sonuclari.xlsx`
- `results/reports/03_co2_ablation_sonuclari.xlsx`
- `results/reports/00_ozet_rapor.xlsx` (summary report)

### Additional Analysis Modules

#### Diurnal Profile Analysis

```bash
python -m preprocessing.daily_profiles
```

#### Pareto Front Analysis

```bash
python -m analysis.pareto_energy_water
```

#### Solution Stability Analysis

```bash
python -m analysis.solution_stability
```

#### Noise Robustness Analysis

```bash
python -m analysis.noise_robustness
```

## How to Execute D-Wave and Classical Annealing Modules

### Classical Simulated Annealing

Classical simulated annealing is the default optimization method and requires no external setup:

```python
from analysis.optimise_sa import run_sa_optimization
from analysis.build_hamiltonian import build_hamiltonian_from_data

# Build Hamiltonian
bqm, J, h = build_hamiltonian_from_data(
    sensor_data={'temperature': 20.0, 'humidity': 60.0},
    co2_penalty=0.0
)

# Optimize
best_sample, best_energy, metadata = run_sa_optimization(bqm, num_reads=1000)
```

### D-Wave Quantum Annealing

**Prerequisites:**
- D-Wave Leap account (free tier available)
- Valid `DWAVE_API_TOKEN` environment variable

**Usage:**

```python
from dwave_annealing.optimise_qanneal import run_quantum_optimization
from analysis.build_hamiltonian import build_hamiltonian_from_data

# Build Hamiltonian
bqm, J, h = build_hamiltonian_from_data(
    sensor_data={'temperature': 20.0, 'humidity': 60.0},
    co2_penalty=0.0
)

# Optimize with quantum annealer
best_sample, best_energy, metadata = run_quantum_optimization(
    bqm, 
    num_reads=1000,
    token=os.getenv('DWAVE_API_TOKEN')
)
```

**Note:** Quantum annealing requires internet connection and consumes D-Wave API credits. Use sparingly for testing.

### Running from Command Line

```bash
# Classical annealing (default in all scripts)
cd analysis/scripts
python 03_experiments_co2_tests.py

# Quantum annealing (requires D-Wave token)
# Modify scripts to set use_quantum=True in co2_sensitivity.py
```

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

1. **Classical Simulated Annealing** (`dwave-neal`): Default method, no external dependencies
2. **Quantum Annealing** (`dwave-system`): Requires D-Wave API credentials

## Troubleshooting

### Common Issues

**Issue: ModuleNotFoundError**
```bash
# Solution: Ensure you're in the project root directory
cd QIM-Greenhouse-Optimization
# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Issue: Dataset not found**
```bash
# Ensure greenhouse_data.xlsx is in data/ directory
ls data/greenhouse_data.xlsx
```

**Issue: D-Wave connection failed**
```bash
# Check API token is set
echo $DWAVE_API_TOKEN

# Verify token is valid
python -c "from dwave.system import DWaveSampler; print(DWaveSampler().solver)"
```

**Issue: Permission denied (Windows)**
- Run PowerShell/Command Prompt as Administrator
- Or ensure write permissions for `results/` directory

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

For questions or issues, please refer to the corresponding author of the article or open an issue on GitHub.

## Acknowledgments

- TÜBİTAK Project 7240482 – "Topraksız Tarıma Yönelik Makine Öğrenimi Tabanlı Akıllı Sistem Çözümleri"
- EasyFlex Smart Greenhouse Automation System
- D-Wave Systems (for quantum annealing capabilities)
