validation = {
    "scenario_name": "validation",
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

CurClim_DynAdapt = {
    "scenario_name": "CurClim_DynAdapt",
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

DryClim_DynAdapt = {
    "scenario_name": "DryClim_DynAdapt",
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

WetClim_DynAdapt = {
    "scenario_name": "WetClim_DynAdapt",
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

CurClim_NoAdapt = {
    "scenario_name": "CurClim_NoAdapt",
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

DryClim_NoAdapt = {
    "scenario_name": "DryClim_NoAdapt",
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

WetClim_NoAdapt = {
    "scenario_name": "WetClim_NoAdapt",
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

CurClim_FixAdaptWeir = {
    "scenario_name": "CurClim_FixAdaptWeir",
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

CurClim_FixAdaptChannel = {
    "scenario_name": "CurClim_FixAdaptChannel",
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