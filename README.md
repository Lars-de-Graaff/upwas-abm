# upwas-abm

# UPWAS ABM: Agent-Based Model for Water Storage Adaptation

An agent-based model (ABM) implementation for studying farmer adaptation strategies in water management systems. UPWAS (Upscaling Water Storage) integrates hydrological modeling with behavioral decision-making to simulate agricultural water storage decisions under different climate scenarios.

## 🔬 Overview

The UPWAS ABM couples the [WALRUS hydrological model](walrus_abm_version.py) with farmer agents that make adaptive decisions about water storage measures (weirs and reduced channel depth) based on Protection Motivation Theory (PMT). The model simulates socio-hydrological dynamics across agricultural landscapes under various climate conditions.

### Key Features

- **Hydrological Modeling**: Integration with WALRUS for realistic water dynamics
- **Behavioral Modeling**: Agent decision-making based on Protection Motivation Theory
- **Climate Scenarios**: Support for historical and future climate projections

## 📁 Repository Structure
├── upwas_model.py # Main ABM model implementation ├── upwas_agents.py # Farmer agent definitions ├── upwas_scenarios.py # Scenario configurations ├── walrus_abm_version.py # WALRUS hydrological model integration ├── upwas_run.py # Single scenario execution ├── upwas_batch_run.py # Basic batch execution ├── upwas_batch_run_with_tracking.py # Enhanced batch with monitoring ├── requirements.txt # Python dependencies ├── HPC_DEPLOYMENT_GUIDE.md # HPC deployment instructions ├── data/ # Input data directory │ ├── climate/ # Climate forcing data │ └── dem/ # Digital elevation models └── output/ # Model outputs
