# upwas-abm

# Agent-Based Model for drought adaptive behaviour 

An agent-based model (ABM) implementation for studying farmer adaptation strategies in water management systems. The model integrates hydrological modeling with behavioral decision-making to simulate agricultural adaptation decisions under different climate scenarios.

## Overview

The ABM couples the [WALRUS hydrological model](walrus_abm_version.py) with farmer agents that make adaptive decisions about drought adaptation measures (weirs and reduced channel depth) based on Protection Motivation Theory (PMT). The model simulates socio-hydrological dynamics across agricultural landscapes under various climate conditions.

### Key Features

- **Hydrological Modeling**: Integration with WALRUS for realistic water dynamics
- **Behavioural Modeling**: Agent decision-making based on Protection Motivation Theory
- **Climate Scenarios**: Support for historical and future climate projections

## Repository Structure

```
├── upwas_model.py               # Main ABM model implementation
├── upwas_agents.py              # Farmer agent definitions
├── upwas_scenarios.py           # Scenario configurations
├── walrus_abm_version.py        # WALRUS hydrological model integration
├── upwas_run.py                 # Single scenario execution
├── requirements.txt             # Python dependencies
├── data/                        # Input data directory
│   ├── climate/                 # Climate forcing data
│   └── dem/                     # Digital elevation models
└── output/                      # Model outputs
```
**Available scenarios:**
- `validation`: Validation run
- `CurClim_FixAdaptWeir`: scenario with a fixed number of weirs installed, no reduced channels
- `CurClim_FixAdaptChannel`: scenario with a fixed number of reduced channels, no weirs
- `CurClim_DynAdapt`: Dynamic Adaptation scanario with current climate conditions
- `DryClim_DynAdapt`: Dynamic Adaptation scenario with dry climate conditions
- `WetClim_DynAdapt`: Dynamic Adaptation scenario with wet climate conditions
- `CurClim_NoAdapt`: No adaptation scanario with current climate conditions
- `DryClim_NoAdapt`: No adaptation scenario with dry climate conditions
- `WetClim_NoAdapt`: No adaptation scenario with wet climate conditions
- `CurClim_MaxAdapt`: Maximum adaptation scanario with current climate conditions
- `DryClim_MaxAdapt`: Maximum adaptation scenario with dry climate conditions
- `WetClim_MaxAdapt`: Maximum adaptation scenario with wet climate conditions

### Funding
This research is part of the project ‘Upscaling private and collective water storage for robust agricultural systems: Potentials, possibilities and challenges’ (UPWAS; project number KICH1.LWV02.20.006) of the research programme ‘Climate-robust production systems and water management’ (KIC) which is financed by the Dutch Research Council (NWO).
