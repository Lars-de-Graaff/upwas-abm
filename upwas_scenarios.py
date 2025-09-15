historic = {
    "scenario_name": "historic",
    "climate" : "2010-2024",
    "total_runs": 40,
    "start_date": 20100101,
    "end_date": 20250101,
    "cell_size": 800,
    "grid_width": 20,
    "grid_height": 20,
    "base_seed": 186,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": True,
    "initial_weir" : 0.01,
    "fixed_reduced_channels": None,
    "no_channel": False
}

proactive = {
    "scenario_name": "proactive",
    "climate" : "2010-2024",
    "total_runs": 40,
    "start_date": 20100101,
    "end_date": 20250101,
    "cell_size": 800,
    "grid_width": 20,
    "grid_height": 20,
    "base_seed": 286,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": True,
    "initial_weir" : 0.25,
    "fixed_reduced_channels": None,
    "no_channel": False
}

future_dry = {
    "scenario_name": "future_dry",
    "climate" : "2050_hd_ens1",
    "total_runs": 40,
    "start_date": 20500101,
    "end_date": 20650101,
    "cell_size": 800,
    "grid_width": 20,
    "grid_height": 20,
    "base_seed": 386,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": True,
    "initial_weir" : 0.25,
    "fixed_reduced_channels": None,
    "no_channel": False
}

future_wet = {
    "scenario_name": "future_wet",
    "climate" : "2050_hn_ens1",
    "total_runs": 40,
    "start_date": 20500101,
    "end_date": 20650101,
    "cell_size": 800,
    "grid_width": 20,
    "grid_height": 20,
    "base_seed": 386,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": True,
    "initial_weir" : 0.25,
    "fixed_reduced_channels": None,
    "no_channel": False
}

reference_2010_2024 = {
    "scenario_name": "reference_2010_2024",
    "climate" : "2010-2024",
    "total_runs": 40,
    "start_date": 20100101,
    "end_date": 20250101,
    "cell_size": 800,
    "grid_width": 20,
    "grid_height": 20,
    "base_seed": 186,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": False,
    "initial_weir" : 0,
    "fixed_reduced_channels": None,
    "no_channel": False
}

reference_2050_dry = {
    "scenario_name": "reference_2050_dry",
    "climate" : "2050_hd_ens1",
    "total_runs": 40,
    "start_date": 20500101,
    "end_date": 20650101,
    "cell_size": 800,
    "grid_width": 20,
    "grid_height": 20,
    "base_seed": 386,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": False,
    "initial_weir" : 0,
    "fixed_reduced_channels": None,
    "no_channel": False
}

reference_2050_wet = {
    "scenario_name": "reference_2050_wet",
    "climate" : "2050_hn_ens1",
    "total_runs": 40,
    "start_date": 20500101,
    "end_date": 20650101,
    "cell_size": 800,
    "grid_width": 20,
    "grid_height": 20,
    "base_seed": 386,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": False,
    "initial_weir" : 0,
    "fixed_reduced_channels": None,
    "no_channel": False
}

fixed_weirs = {
    "scenario_name": "fixed_weirs",
    "climate" : "2010-2024",
    "total_runs": 40,
    "start_date": 20100101,
    "end_date": 20250101,
    "cell_size": 800,
    "grid_width": 20,
    "grid_height": 20,
    "base_seed": 186,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": False,
    "initial_weir" : 0.5,
    "fixed_reduced_channels": None,
    "no_channel": True
}

fixed_reduced_channels = {
    "scenario_name": "fixed_reduced_channels",
    "climate" : "2010-2024",
    "total_runs": 40,
    "start_date": 20100101,
    "end_date": 20250101,
    "cell_size": 800,
    "grid_width": 20,
    "grid_height": 20,
    "base_seed": 186,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": False,
    "initial_weir" : 0,
    "fixed_reduced_channels": 0.5,
    "no_channel": False
}

test_fast = {
    "scenario_name": "test_fast",
    "climate" : "2010-2024",
    "total_runs": 1,
    "start_date": 20100101,
    "end_date": 20110101,
    "cell_size": 800,
    "grid_width": 3,
    "grid_height": 3,
    "base_seed": 186,
    "alpha": 0.6,
    "gamma": 0.5,
    "intention2adapt" : 45,
    "implement_measure_on": True,
    "initial_weir" : 1,
    "fixed_reduced_channels": None,
    "no_channel": True
}