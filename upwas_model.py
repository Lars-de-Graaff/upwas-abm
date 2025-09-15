import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from mesa import Model 
from mesa.space import SingleGrid, PropertyLayer
from mesa.datacollection import DataCollector
import upwas_agents



################# Define model class #################
class UpwasModel(Model):
    def __init__(self, width, height, start_date, end_date, cell_size, dem=None, seed=None, alpha=0.5, gamma=0.5,
                  initial_weir=0, intention2adapt=45, climate="2010-2024", implement_measure_on=True, fixed_reduced_channels=None,no_channel=False):
        super().__init__(seed=seed)
        self.grid = SingleGrid(width, height, False)
        self.alpha = alpha
        self.gamma = gamma
        self.initial_weir = initial_weir # chance that the farmer has a weir installed at the start of the model run
        self.intention2adapt = intention2adapt
        self.dem = dem
        self.climate = climate
        self.implement_measure_on = implement_measure_on
        self.no_channel = no_channel
        self.fixed_reduced_channels = fixed_reduced_channels  # Fixed percentage of farmer with reduced channel depth
        self.date = (datetime.strptime(str(start_date), '%Y%m%d'))
        self.timestep_length = timedelta(days=1)
        self.grid.add_property_layer(PropertyLayer("elevation", width, height, 0, dtype=int))
        self._init_property_layers(dem)
        print(f"Model initialized with {width}x{height} grid, start date {start_date}, end date {end_date}, cell size {cell_size}, and climate {climate}")
        
        #TODO model data collector for avarage groundwater levels ect.
        self.datacollector_daily = DataCollector(agent_reporters={"Farmer":"unique_id","Date":"date","Groundwater depth": "gw_level", "Surface water level": "sw_level",
                                                             "hSmin_rel": "hSmin_rel","Surface water relative": "sw_level_rel", "Discharge": "discharge"}) #"Age":"age","P":"p",
        
        self.datacollector_yearly = DataCollector(agent_reporters={"Farmer":"unique_id","Position": "pos", "Date":"date", "Weir":"weir_installed","Channel reduced": "channel_reduced","Threat appraisal": "threat_appraisal",
                                                             "Coping appraisal weir": "coping_appraisal_weir","Coping appraisal channel": "coping_appraisal_channel","Intention weir": "intention_to_adapt_weir",
                                                             "Intention channel": "intention_to_adapt_channel", "Operation strategy": "operation_strategy","Drought days": "drought_stress",
                                                             "Wet days": "wet_damage", "Mean drought":"mean_drought_stress"})
        
        self.datacollector_farmer = DataCollector(agent_reporters={"Farmer":"unique_id","Position": "pos", "Walrus":"soil_pars", "Elevation": "elevation", "Neighbourhood radius": "neighbourhood_radius",
                                                                    "Successor":"successor", "Threshold dry": "threshold_dry", "Threshold wet": "threshold_wet", "Alpha": "alpha", "conductivity":"k"})

        # Add agents to every cell
        for i in range(width * height):
            agent = upwas_agents.Farmer(i, self, start_date, end_date, cell_size, self.alpha, self.gamma, self.intention2adapt,
                                         self.initial_weir, self.climate, self.implement_measure_on, self.fixed_reduced_channels, self.no_channel) 
            self.grid.move_to_empty(agent)
            self.agents.do("define_surroudings")
        self.datacollector_farmer.collect(self)
        self.datacollector_yearly.collect(self)  # Collect initial state variables
        print(f'initialised {len(self.agents)} agents')

    def _init_property_layers(self, dem):
        if dem is None:
            self.grid.properties["elevation"].data[:] = np.random.randint(20, 30, self.grid.properties["elevation"].data.shape)
        else:
            elevation_data = pd.read_csv(dem, delimiter=',', header=None)
            self.grid.properties["elevation"].data[:] = elevation_data #np.random.randint(10, 40, elevation_data.shape)


    def collect_variables(self):
        self.datacollector_daily.collect(self)
        if self.date.month == 12 and self.date.day == 31:
            self.datacollector_yearly.collect(self)
        self.date = self.date + self.timestep_length
   

    def step(self):
        self.collect_variables()
        self.agents.do("sociohydrology")
        self.agents.do("exchange_and_proceed")
