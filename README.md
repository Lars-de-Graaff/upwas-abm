# upwas-abm

# UPWAS ABM: Agent-Based Model for Water Storage Adaptation

An agent-based model (ABM) implementation for studying farmer adaptation strategies in water management systems. UPWAS (Upscaling Water Storage) integrates hydrological modeling with behavioral decision-making to simulate agricultural water storage decisions under different climate scenarios.

## Overview

The UPWAS ABM couples the [WALRUS hydrological model](walrus_abm_version.py) with farmer agents that make adaptive decisions about water storage measures (weirs and reduced channel depth) based on Protection Motivation Theory (PMT). The model simulates socio-hydrological dynamics across agricultural landscapes under various climate conditions.

### Key Features

- **Hydrological Modeling**: Integration with WALRUS for realistic water dynamics
- **Behavioral Modeling**: Agent decision-making based on Protection Motivation Theory
- **Climate Scenarios**: Support for historical and future climate projections

## 📁 Repository Structure

```
├── upwas_model.py                      # Main ABM model implementation
├── upwas_agents.py                     # Farmer agent definitions
├── upwas_scenarios.py                  # Scenario configurations
├── walrus_abm_version.py              # WALRUS hydrological model integration
├── upwas_run.py                       # Single scenario execution
├── requirements.txt                   # Python dependencies
├── HPC_DEPLOYMENT_GUIDE.md           # HPC deployment instructions
├── data/                             # Input data directory
│   ├── climate/                      # Climate forcing data
│   └── dem/                         # Digital elevation models
└── output/                          # Model outputs
```
**Available scenarios:**
- `validation`: Validation run
- `measure_only_weirs`: scenario with a static number of weirs installed, no reduced channels
- `measure_only_reduced_channels`: scenario with a static number of reduced channels, no weirs
- `dynamic_current_climate`: Adaptive bahaviour scanario with current climate conditions
- `dynamic_dry_future`: Adaptive bahaviour scenario with dry climate conditions
- `dynamic_wet_future`: Adaptive bahaviour scenario with wet climate conditions
- `static_current_climate`: Static scanario with current climate conditions
- `static_dry_future`: Static scenario with dry climate conditions
- `static_wet_future`: Static scenario with wet climate conditions

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/upwas-abm.git
   cd upwas-abm
   ```

2. **Set up environment using conda (recommended):**
   ```bash
   conda env create -f environment.yml
   conda activate upwas-abm
   ```

   **Or using pip:**
   ```bash
   pip install -r requirements.txt
