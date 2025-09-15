from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import os
import upwas_model as upwas
import upwas_scenarios

scenario = upwas_scenarios.test_fast #None # upwas_scenarios.{scenario_name} (Global variable to hold the scenario configuration)

if scenario is None:
    config = {
        "total_runs": 1,
        "scenario_name": "test_run",
        "model": "upwas_fixed_reduced_channels",#"upwas_standard",  # Default model, can be changed to "upwas_fixed_reduced_channels"
        "climate" : "2010-2024",
        "start_date": 20100101,
        "end_date": 20150101,
        "cell_size": 800,
        "grid_width": 3, # options: 3, 5, and 20
        "grid_height": 3, # options: 3, 5, and 20
        "base_seed": 186,
        "alpha": 0.6,
        "gamma": 0.5,
        "intention2adapt" : 45,
        "implement_measure_on": True,
        "initial_weir" : 0.01,
        "fixed_reduced_channels": None
    }
else:
    config = scenario
print(f"Running scenario: {config['scenario_name']} with {config}")

def run_model(config, run=1):
    output_dir = f"output/{config['scenario_name']}"
    dem_file = f"data/dem/dem_{config['grid_width']}x{config['grid_height']}.csv"
    seed = config["base_seed"]
    np.random.seed(seed)
    start_time = time.time()
    steps = (datetime.strptime(str(config["end_date"]), "%Y%m%d") - datetime.strptime(str(config["start_date"]), "%Y%m%d")).days
    model_run = f"{config['scenario_name']}_scenario_{config['climate']}_{config['grid_width']}x{config['grid_height']}_run_{run}"
    
    # Create model with appropriate parameters
    model_params = {
        "width": config["grid_width"],
        "height": config["grid_height"], 
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "cell_size": config["cell_size"],
        "dem": dem_file,
        "seed": seed,
        "alpha": config["alpha"],
        "gamma": config["gamma"],
        "initial_weir": config["initial_weir"],
        "intention2adapt": config["intention2adapt"],
        "climate": config["climate"],
        "implement_measure_on": config["implement_measure_on"],
        "fixed_reduced_channels": config.get("fixed_reduced_channels", None),
        "no_channel": config.get("no_channel", False)   
    }
     
    model = upwas.UpwasModel(**model_params)

    for _ in tqdm(range(steps), desc=f"Run {run} progress"):
        model.step()

    os.makedirs(output_dir, exist_ok=True)
    model.datacollector_daily.get_agent_vars_dataframe().to_csv(f"{output_dir}/daily_data_run_{run:02d}.csv", index=False, sep=";")
    model.datacollector_yearly.get_agent_vars_dataframe().to_csv(f"{output_dir}/yearly_data_run_{run:02d}.csv", index=False, sep=";")
    model.datacollector_farmer.get_agent_vars_dataframe().to_csv(f"{output_dir}/farmer_data_run_{run:02d}.csv", index=False, sep=";")

    run_time = round(time.time() - start_time, 2)
    # Save run parameters
    params = {
        "start_date": [config["start_date"]],
        "end_date": [config["end_date"]],
        "steps": [steps],
        "time_taken": [round(run_time, 2)],
        "seed": [seed],
        "name_run": [model_run],
        "num_agents": [len(model.agents)],
        "run_id": [run],
        "alpha": [config["alpha"]],
        "gamma": [config["gamma"]],
        "initial_weir": [config["initial_weir"]],
        "intention2adapt": [config["intention2adapt"]],
        "climate": [config["climate"]],
        "scenario_name": [config["scenario_name"]],
        "implement_measure_on": [config["implement_measure_on"]],
        "fixed_reduced_channels": [config.get("fixed_reduced_channels", None)],
        "no_weirs": [config.get("no_weirs", False)],
        "grid_width": [config["grid_width"]],
        "grid_height": [config["grid_height"]],
        "cell_size": [config["cell_size"]],
    }
    # Save parameters to CSV
    pd.DataFrame(params).to_csv(f"{output_dir}/model_parameters_run_{run:02d}.csv", index=False, sep=";")

def main():
    for i in range(config['total_runs']):
        run_model(config, i+1)

if __name__ == "__main__":
    main()

