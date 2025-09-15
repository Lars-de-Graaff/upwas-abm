import numpy as np
import pandas as pd
import calendar
from datetime import datetime, timedelta
from mesa import Agent
from scipy.interpolate import interp1d
from walrus_abm_version import WALRUS_step_numba, WALRUS_array_to_dict, WALRUS_dict_to_array,WALRUS_select_dates,WALRUS_preprocessing, WALRUS_prepare_loop, WALRUS_loop_numba, p_num


class Farmer(Agent):
    def __init__(self, unique_id, model, start_date, end_date, cell_size, alpha, gamma, intention2adapt,
                 initial_weir, climate, implement_measure_on, fixed_reduced_channels, no_channel):
        super().__init__(model)
        self.model = model
        self.start_date = start_date
        self.end_date = end_date
        self.cell_size = cell_size
        self.alpha = alpha  # bahvioural factor for intention to adapt
        self.gamma = gamma  # behavioral factor for coping appraisal
        self.beta = 1 - self.alpha  # Ensure alpha + beta = 1
        self.delta = 1 - self.gamma  # Ensure gamma + delta = 1
        self.intention2adapt = intention2adapt
        self.initial_weir = initial_weir # chance that the farmer has a weir installed at the start of the model run
        self.climate = climate
        self.implement_measure_on = implement_measure_on
        self.no_channel = no_channel  # If True, no weirs are installed
        self.fixed_reduced_channels = fixed_reduced_channels  # Fixed percentage of farmer with reduced channel depth
        self._initiate_attributes()
        self._initiate_hydrology()


################# initialise farmers attributes #################
    def _initiate_attributes(self):
        '''
        Initiate attributes of agents
        '''
        self.date = (datetime.strptime(str(self.start_date), '%Y%m%d'))
        self.timestep_length = timedelta(days=1)
        # Farmer characteristics 
        self.age = np.random.normal(45, 14) #np.random.randint(20, 65)
        self.successor = float(np.random.choice([0, 0.5, 1])) # 0=No, 0.5=maybe, 1=Yes
        self.neighbourhood_radius = np.random.randint(1,4) # radius of surrounding cells to compare self with neighbours
        
        self.threshold_dry = np.random.randint(1900,2100) 
        self.threshold_wet = np.random.randint(400,700) 
        
        self.wet_damage = 0 
        self.drought_stress = 0
        self.mean_drought_stress = np.random.randint(10,40)
        
        # Adaptation measure variables
        if np.random.random() <= self.initial_weir: # chance that the farmer has a weir installed
            self.weir = 1 # 0=No, 1=Yes
            self.weir_installed = self.date.year #year of installation
            self.operation_strategy = 1 #fixed operation strategy
        else:
            self.weir = 0 # 0=No, 1=Yes
            self.weir_installed = None
            self.operation_strategy = 0 # 0=None, 1=Fixed, 2=Flexible
        
        if self.fixed_reduced_channels is not None: # If the model run is with a fixed initial number of reduced channels
            if np.random.random() <= self.fixed_reduced_channels:
                self.channel = 1
                self.channel_reduced = self.date.year #year of installation
            else:
                self.channel = 0 # 0=No, 1=Yes
                self.channel_reduced = None
        
        elif self.no_channel == True: # If the model run is with no channels
            self.channel = 0
            self.channel_reduced = None

        else: # If the model run is with a variable number of reduced channels, based on initial number of weirs
            if self.weir == 1 and np.random.random() <= (self.initial_weir/2): # chance that the farmer has a channel depth reduction installed
                self.channel = 1 # 0=No, 1=Yes  
                self.channel_reduced = self.date.year #year of installation
            else:            
                self.channel = 0 # 0=No, 1=Yes
                self.channel_reduced = None
        
        self.forecast_count = np.random.randint(1,8) #Internal counter for the agent to use the weather forecast

        #pmt components
        self.perceived_effectiveness_weir = np.random.uniform(0.3, 0.7)
        self.perceived_effectiveness_channel = np.random.uniform(0.3, 0.7)
        self.knowledge = np.random.uniform(0, 0.2) # Knowledge about the effectiveness of the measures
        self.self_efficacy = min(0.5 * (((65 - self.age) / 45) + (self.successor * 0.1)) + self.knowledge, 1) #0.5 * (neighbour_with_weir / total_neigbours) + self.knowledge, 1)

        self.threat_appraisal = np.random.uniform(0.3, 0.7)
        self.coping_appraisal_weir = min((self.gamma * self.perceived_effectiveness_weir + self.delta * self.self_efficacy), 1) 
        self.coping_appraisal_channel = min((self.gamma * self.perceived_effectiveness_channel + self.delta * self.self_efficacy), 1)

        self.intention_to_adapt_weir = self.alpha * self.threat_appraisal + self.beta * self.coping_appraisal_weir
        self.intention_to_adapt_channel = self.alpha * self.threat_appraisal + self.beta * self.coping_appraisal_channel

    def _initiate_hydrology(self):    
        '''
        This functions initiate the WALRUS model for each farmer agent, based on soil properties and a warmup period of 1 year.
        The hydrological WALRUS data is stored in hydro_stat. 
        Input data:
            start_date: start date of the model, for the warmup period an additional year of data is required
            end_date: end date of the model
        Returns:
            hydro_stat dataframe with WALRUS output
            Initial values for GW level, SM deficit and SW level
            Initial input data for WALRUS run
        '''
        #### Initiate farmer's hydrology #####
        self.PEQ = pd.read_csv(f"data/climate/PEQ_Hupsel_day_{self.climate}.csv", delimiter=",", decimal='.')
                
        self.soil_pars = {"cW":np.random.randint(175,225), "cV": 10, "cG": np.random.randint(15e6, 20e6),  "cQ": np.random.randint(10,15), 
               "dG0": 1000, "cD": np.random.randint(1200,2200), "aS": 0.01, "st": 1} #1 = loamy sand
        self.soil_pars['cS'] = self.soil_pars['cD']/300    
        
        if self.channel == 1:
            self.soil_pars['cD'] = np.random.uniform(0.50,0.75) * self.soil_pars['cD']
            self.soil_pars['cS'] = self.soil_pars['cD']/300
            self.hSmin = 0.5 * self.soil_pars["cD"] # set the weir level to the new channel depth
        
        self.k = np.random.uniform(0.5e3, 1.5e3) # Hydrological connectivity 
        self.hSmin = 0
        self.inflow_surface = 0
        self.inflow_ground = 0 
        
        self.forc = WALRUS_select_dates(self.PEQ, (self.start_date - 10000), self.end_date) 
        self.output_date = WALRUS_preprocessing(self.forc, 1)

        # Interpolate the forc data to create cumulative functions for P, ETpot, fXG and fXS
        self.func_P = interp1d(np.pad(self.forc['date'].values, (0,1), 'constant'), np.cumsum(np.pad(self.forc['P'].values, (1,0), 'constant')), kind='linear', fill_value="extrapolate")
        self.func_ETpot = interp1d(np.pad(self.forc['date'].values, (0,1), 'constant'), np.cumsum(np.pad(self.forc['ETpot'].values, (1,0), 'constant')), kind='linear', fill_value="extrapolate")
        self.func_fXG = interp1d(np.pad(self.forc['date'].values, (0,1), 'constant'), np.cumsum(np.pad(self.forc['fXG'].values, (1,0), 'constant')), kind='linear', fill_value="extrapolate")
        self.func_fXS = interp1d(np.pad(self.forc['date'].values, (0,1), 'constant'), np.cumsum(np.pad(self.forc['fXS'].values, (1,0), 'constant')), kind='linear', fill_value="extrapolate")
        
        func_Qobs = interp1d(np.pad(self.forc['date'].values, (0,1), 'constant'), np.cumsum(np.pad(self.forc["Q"].values,(1,0), 'constant')), kind='linear', fill_value="extrapolate")
        func_hSmin = interp1d(np.pad(self.forc['date'].values, (0,1), 'constant'), np.pad(self.forc['hSmin'].values,(1,0), 'edge'), kind='linear', fill_value='extrapolate')

        # Create a DataFrame for the initial states
        self.hydro_stat = pd.DataFrame(index=range(len(self.output_date)), columns=["ETact", "Q", "fGS", "fQS", "dV", "dVeq", "dG", "hQ", "hS", "W", "dt_ok", "hSmin", "fXS", "fXG"])
        
        # Check if the year before the start date (warmup period) is a leap year
        self.end_warmup = 365 if calendar.isleap(self.date.year - 1) else 364

        warmup = self.output_date.iloc[0:self.end_warmup+1]

        input_walrus_init = WALRUS_prepare_loop(self.soil_pars, warmup, self.hydro_stat, func_Qobs, func_hSmin)
        self.hydro_stat = WALRUS_loop_numba(self.soil_pars, warmup, self.hydro_stat, input_walrus_init, self.func_P, self.func_ETpot, self.func_fXG, self.func_fXS)
      
        self.input_walrus = self.hydro_stat.iloc[self.end_warmup]
        self.sm_deficit = self.hydro_stat.iloc[self.end_warmup]["dV"]
        self.gw_level = self.hydro_stat.iloc[self.end_warmup]["dG"]
        self.sw_level = self.hydro_stat.iloc[self.end_warmup]["hS"]
        self.sw_level_rel = self.sw_level / self.soil_pars["cD"] # convert surface water level to relative water level (0-1) of the total channel depth

################# Step function #################
    def sociohydrology(self):
        if self.date.month == 12 and self.date.day == 15:
            self.update_pmt()
            if self.implement_measure_on == True:
                self.implement_measure()
        self.update_hydrology()
        self.damage()
        self.operate_weir()
        
    def exchange_and_proceed(self):
        self.exchange_groundwater()
        self.proceed_day_of_year()


    def define_surroudings(self): #only for the first step
        self.elevation = self.model.grid.properties['elevation'].data[self.pos]
        self.direct_neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=True, radius=1) 
        self.neighbourhood_neighbours = self.model.grid.get_neighbors(self.pos, moore=False, include_center=False, radius=self.neighbourhood_radius)
        self.total_neigbours = self.neighbourhood_radius ** 2 + (self.neighbourhood_radius + 1)**2 - 1
        

################## Protection Motivation Theory #################
    def update_pmt(self):
        self.update_threat_appraisal()
        self.update_coping_appraisal()
        
        #calc intention to adapt
        self.intention_to_adapt_weir = self.alpha * self.threat_appraisal + self.beta * self.coping_appraisal_weir
        self.intention_to_adapt_channel = self.alpha * self.threat_appraisal + self.beta * self.coping_appraisal_channel


        
    def update_threat_appraisal(self):
        """
        This function updates the threat appraisal of the farmer agent.
        """
        if self.drought_stress > self.mean_drought_stress:
            damage_own = (1/183) * (self.drought_stress - self.mean_drought_stress)
            self.threat_appraisal = self.threat_appraisal + (damage_own * self.threat_appraisal) #Update threat appraisal based on damage function
            self.threat_appraisal = min(self.threat_appraisal, 1)
        else:
            self.threat_appraisal = self.threat_appraisal * 0.875 # Threat appraisal decreases with 12.5% every year when there is no impact of drought
        
        if self.wet_damage > 20:
            self.threat_appraisal = self.threat_appraisal * 0.875 # decrease threat appraisal with 12.5% (too wet, no focus on drought impact)
        
        #Update mean drought stress of the farmer
        self.mean_drought_stress = (0.9 * self.mean_drought_stress) + (0.1 * self.drought_stress) # Mean drought stress is calculated based on 90% of existing mean and 10% on drought stress of this year

    def update_coping_appraisal(self): 
        """
        This function updates the coping appraisal of the farmer agent. 
        The coping appraisal is based on the perceived effectiveness of the measures, the self-efficacy and the perceived cost of the measures.
        """
        ###### Update coping appraisal weir ######
        neighbour_with_weir = 0
        neighbours_with_channel = 0

        ###### Too wet situations ######
        if self.wet_damage > 20: # decrease coping appraisal if the farmer has experienced too many wet conditions
            self.perceived_effectiveness_weir = self.perceived_effectiveness_weir * 0.90 # decrease coping appraisal weirs
            self.perceived_effectiveness_channel = self.perceived_effectiveness_channel * 0.85 # decrease coping appraisal channel
        else:
            for neighbour in self.neighbourhood_neighbours:
                if neighbour.weir == 1:
                    neighbour_with_weir += 1
                if neighbour.channel == 1:
                    neighbours_with_channel += 1

                if self.weir == 0 and neighbour.weir == 1:                
                    damage_compare = (1/183) * (self.drought_stress - neighbour.drought_stress)
                    self.perceived_effectiveness_weir = self.perceived_effectiveness_weir + (damage_compare * self.perceived_effectiveness_weir)# increase coping appraisal weirs
                    
                    if self.wet_damage > 10 and self.wet_damage < neighbour.wet_damage:
                        self.perceived_effectiveness_weir = self.perceived_effectiveness_weir * 0.93 # decrease coping appraisal weirs
                elif self.weir == 1 and neighbour.weir == 0:
                    damage_compare = (1/183) * (neighbour.drought_stress - self.drought_stress)
                    self.perceived_effectiveness_weir = self.perceived_effectiveness_weir + (damage_compare * self.perceived_effectiveness_weir)

                if self.channel == 0 and neighbour.channel == 1:
                    damage_compare = (1/183) * (self.drought_stress - neighbour.drought_stress)
                    self.perceived_effectiveness_channel = self.perceived_effectiveness_channel + (damage_compare * self.perceived_effectiveness_channel) # increase coping appraisal weirs
                
                    if self.wet_damage > 10 and self.wet_damage < neighbour.wet_damage:
                        self.perceived_effectiveness_channel = self.perceived_effectiveness_channel * 0.93 # decrease coping appraisal weirs
                elif self.channel == 1 and neighbour.channel == 0:
                    damage_compare = (1/183) * (neighbour.drought_stress - self.drought_stress)
                    self.perceived_effectiveness_channel = self.perceived_effectiveness_channel + (damage_compare * self.perceived_effectiveness_channel)


        ###### Influence from waterboard ######
        if self.weir == 0 and self.channel == 0 and np.random.random() <= 0.05: # 5% chance that the farmer will increase the coping appraisal for weirs when waterboards promote the installation of weirs
            self.perceived_effectiveness_weir = self.perceived_effectiveness_weir * 1.15
            self.perceived_effectiveness_channel = self.perceived_effectiveness_channel * 1.15
            self.knowledge += 0.10
        elif self.weir == 1 and self.channel == 0 and np.random.random() <= 0.10: # 10% chance that the farmer will increase the coping appraisal for channels when waterboards promote the reduction of channel depth
            self.perceived_effectiveness_channel = self.perceived_effectiveness_channel * 1.15
            self.knowledge += 0.10

        if self.date.year >= 2018 and np.random.random() <= 0.75:
            self.perceived_cost = 0
        elif self.date.year < 2018 and np.random.random() <= 0.25:
            self.perceived_cost = 0
        else:
            self.perceived_cost = 1

        self.knowledge = min(self.knowledge, 1) # knowledge is between 0 and 1
        self.self_efficacy = min(0.5 * (((65 - (self.age - self.successor*20)) / 45)) + 0.5 * (min(((neighbour_with_weir + neighbours_with_channel) / self.total_neigbours + self.knowledge),1)), 1)

        self.coping_appraisal_weir = min((self.gamma * self.perceived_effectiveness_weir + self.delta * self.self_efficacy), 1) 
        self.coping_appraisal_channel = min((self.gamma * self.perceived_effectiveness_channel + self.delta * self.self_efficacy), 1)



################# implement measure (individually) #################
    def implement_measure(self):
        '''
        This function implements the measures of the farmer agent. The farmer agent can choose to implement a weir or a channel depth reduction.
        '''
        if self.weir == 0 and self.perceived_cost == 0:
            likelihood_to_adapt_weir = ( 1 -  ( ( 1 - self.intention_to_adapt_weir ) ** ( 1 / self.intention2adapt ) ) ) 
            
            if likelihood_to_adapt_weir > np.random.random():
                self.weir = 1
                self.operation_strategy = 1 #fixed operation strategy
                self.weir_installed = self.date.year

        elif self.channel == 0 and self.weir == 1 and self.perceived_cost == 0:
            likelihood_to_adapt_channel = ( 1 -  ( ( 1 - self.intention_to_adapt_channel ) ** ( 1 / self.intention2adapt ) ) )

            if likelihood_to_adapt_channel > np.random.random():
                self.soil_pars["cD"] = np.random.uniform(0.50,0.75) * self.soil_pars["cD"]
                self.soil_pars['cS'] = self.soil_pars['cD']/300
                self.channel = 1 
                self.channel_reduced = self.date.year
                self.hSmin = 0.5 * self.soil_pars["cD"] # set the weir level to the new channel depth
        
        elif self.weir == 1 and self.operation_strategy == 1:   
            if self.coping_appraisal_weir >= np.random.random(): # if the coping appraisal is higher than the random value, the agent will switch to flexible operation strategy
                self.operation_strategy = 2 #Flexible operation strategy
        
################# operate weirs ##################
    def operate_weir(self): 
        """
        This function operates the weir based on the operation strategy of the farmer agent. The weir level is set based on the operation strategy and the current date.
        The weir level is set to a minimum level (hSmin). The hSmin is set based on the operation strategy and the current date. 
        The maximum water level of hSmin is set to 0.8 * soil_pars["cD"].
        """	       
        if self.operation_strategy == 1:
            self.operate_weir_fixed()   
        elif self.operation_strategy == 2:
            self.operate_weir_flexible()
        # Convert hSmin to relative level (0-1) of the total channel depth
        self.hSmin_rel = self.hSmin / self.soil_pars["cD"] # convert hSmin to relative level (0-1)
        
    def operate_weir_fixed(self):# fixed operration strategy
        if self.date.month <= 2:
            self.hSmin = 0.5 * self.soil_pars["cD"]
        elif self.date.month == 3:
            if self.date.day == 1:
                self.hSmin = 0.25
            elif self.date.day == 5:
                self.hSmin = 0
        elif self.date.month >= 4 and self.date.month <= 5:
            self.hSmin = 0.6 * self.soil_pars["cD"]
        elif self.date.month >= 6 and self.date.month <= 9:
            self.hSmin = 0.8 * self.soil_pars["cD"]
        elif self.date.month >= 10:
            self.hSmin = 0.2 * self.soil_pars["cD"]

    def operate_weir_flexible(self):# flexible operration strategy, e.g. dependent on weather forcast and current groundwater levels
        if self.forecast_count == 7:
            rain = self.rain_forecast_function()
            rain_forecast = rain[0]

            if np.random.random() <= 0.02: # 2% chance that the agent takes an non-informed and random decision or someone else changed the weir without consent
                self.hSmin = np.random.uniform(0,0.8) * self.soil_pars["cD"]

            elif rain_forecast > 100: #extreme rain forecast
                self.hSmin = 0 # the agent will release all water to avoid water loging
            elif self.gw_level >= (self.threshold_dry - 200): 
                self.hSmin = 0.8 * self.soil_pars["cD"] # the agent will increase the weir level to reduce the chance of drought stress
            elif self.gw_level <= (self.threshold_wet + 100):
                self.hSmin = 0 # the agent will release all water to avoid water loging

            elif self.gw_level >= self.threshold_wet + 250: # and self.sw_level < (self.hSmin - 100) and 
                if rain_forecast > 40:
                    self.hSmin -= np.random.uniform(0.1,0.3) * self.soil_pars["cD"]
                else:
                    self.hSmin += np.random.uniform(0.1,0.3) * self.soil_pars["cD"]

            elif self.gw_level < self.threshold_wet + 250:# and self.sw_level < (self.hSmin - 100) and rain_forecast > 40:
                if rain_forecast < 5:
                    self.hSmin += np.random.uniform(0.0,0.1) * self.soil_pars["cD"]
                else:
                    self.hSmin -= np.random.uniform(0.1,0.3) * self.soil_pars["cD"]

        self.hSmin = max(self.hSmin, 0)
        self.hSmin = min(self.hSmin, 0.8*self.soil_pars["cD"])

################### Use weather forecast ###################
    def rain_forecast_function(self):
        start_index = self.timestep
        p = self.PEQ["P"]
        rain_forecast = []
        for i in range(7):
            if float(p.loc[start_index+i]) == 0:
                p_adj = float(p.loc[start_index+i]) + np.random.uniform(0, 2) #This is a random factor to account for the uncertainty in the weather forecast
                rain_forecast.append(p_adj)
            else:
                p_adj = float(p.loc[start_index+i]) * np.random.normal(1, 0.05) #This is a random factor to account for the uncertainty in the weather forecast
                rain_forecast.append(p_adj)
        sum_rain_forecast = sum(rain_forecast)
        p_past = sum(p.loc[start_index-30:start_index])
        
        return sum_rain_forecast, p_past

############ Damage ###############################
    def damage(self):
        if self.date.month >= 4 and self.date.month <= 9: # Low groundwater levels are only relevant during growing season
            if self.gw_level > self.threshold_dry: 
                self.drought_stress += 1
        
        if self.gw_level < self.threshold_wet: # Too wet situations are relevant throughout the year 
            self.wet_damage += 1

################# hydrological fluxes #################
    def update_hydrology(self):
        timestep = self.model.steps + (self.end_warmup + 1)
        self.timestep = timestep
        start_step = self.output_date.loc[timestep-1, 'date']
        end_step = self.output_date.loc[timestep, 'date']
        sums_step = np.zeros(4)
        self.forc['fXG'] = self.inflow_ground
        
        SOIL_PARS_ORDER = ["cW", "cV", "cG", "cQ", "dG0", "cD", "aS", "st", "cS", "b", "psi_ae", "theta_s", "aG"]
        INPUT_WALRUS_ORDER = ["ETact", "Q", "fGS", "fQS", "dV", "dVeq", "dG", "hQ", "hS", "W", "dt_ok", "hSmin", "fXS", "fXG"]

        soil_pars_arr = np.array([self.soil_pars[key] for key in SOIL_PARS_ORDER], dtype=np.float64)
        input_walrus_arr = np.array([self.input_walrus[key] for key in INPUT_WALRUS_ORDER], dtype=np.float64)
        
        # print(f"input_walrus_arr before while: {input_walrus_arr}")
        while start_step < (self.output_date.loc[timestep, 'date']  - p_num["min_timestep"]):
            P_t = self.func_P(end_step) - self.func_P(start_step)
            ETpot_t = self.func_ETpot(end_step) - self.func_ETpot(start_step)
            fXG_t = self.func_fXG(end_step) - self.func_fXG(start_step)
            fXS_t = self.func_fXS(end_step) - self.func_fXS(start_step)
            dt = (end_step - start_step) / 3600  # in hours
            
            step = WALRUS_step_numba(soil_pars_arr, input_walrus_arr, P_t, ETpot_t, fXG_t, fXS_t, self.hSmin, dt)

            step_dict = WALRUS_array_to_dict(step, INPUT_WALRUS_ORDER)

            if step_dict["dt_ok"] == 0 and (int(end_step) - int(start_step)) > p_num["min_timestep"]:
                end_step = (int(start_step) + int(end_step)) / 2
            else:
                start_step = end_step
                end_step = self.output_date.loc[timestep, 'date']
                sums_step += step[0:4]
                self.hydro_stat.iloc[timestep] = pd.Series(step_dict)
                self.input_walrus = step_dict
                input_walrus_arr = WALRUS_dict_to_array(self.input_walrus, INPUT_WALRUS_ORDER)
        
        self.hydro_stat.iloc[timestep] = pd.Series(step_dict)
        self.hydro_stat.iloc[timestep, 0:4] = sums_step

        self.gw_level = self.hydro_stat.iloc[timestep]["dG"]
        self.sw_level = self.hydro_stat.iloc[timestep]["hS"]
        self.sw_level_rel = self.sw_level / self.soil_pars["cD"]
        self.discharge = self.hydro_stat.iloc[timestep]["Q"]
        self.inflow_ground = 0  # Reset inflow from neighbours

    def exchange_groundwater(self): 
        for neighbour in self.direct_neighbours:
            elevation = self.model.grid.properties["elevation"].data[self.pos]
            elevation_neighbour = self.model.grid.properties["elevation"].data[neighbour.pos]
            if self.gw_level + elevation < neighbour.gw_level + elevation_neighbour:
                ground_discharge = self.k * (((neighbour.gw_level + elevation_neighbour ) - (self.gw_level + elevation))/(self.cell_size*1000)) #Darcy's Law (specific discharge: q = -K * dh/dl) 
                neighbour.inflow_ground += ground_discharge
                self.inflow_ground -= ground_discharge #outcome darcy
        
 
################# store variables and day of the year #################
    def proceed_day_of_year(self):
        if self.date.month == 1 and self.date.day == 1:
            self.drought_stress = 0
            self.wet_damage = 0
            self.age += 1
        self.date = self.date + self.timestep_length
        if self.forecast_count == 7:
            self.forecast_count = 0
        else:
            self.forecast_count += 1
        
        if self.age >= 45 and self.successor == 0.5:
            self.successor  = np.random.randint(2) # the farmer will choose a successor or not

        # Change the farmer when retired
        if self.age > 65 and self.successor == 1:
            self.age = 20
            self.successor = float(np.random.choice([0, 0.5, 1]))
        
        elif self.age > 65 and self.successor == 0: # The farm will be bought by a new farmer
            self.age = np.random.randint(20, 55) 
            self.successor = float(np.random.choice([0, 0.5, 1])) # 0=No, 0.5=maybe, 1=Yes
            
            self.perceived_effectiveness_weir = np.random.uniform(0.1, 0.9)
            self.perceived_effectiveness_channel = np.random.uniform(0.1, 0.9)
            self.knowledge = np.random.uniform(0, 0.2) # Knowledge about the effectiveness of the measures
            self.self_efficacy = min(0.5 * (((65 - self.age) / 45) + (self.successor * 0.1)) + self.knowledge, 1) #0.5 * (neighbour_with_weir / total_neigbours) + self.knowledge, 1)

            self.threat_appraisal = np.random.uniform(0.1, 0.9)
            self.coping_appraisal_weir = min((0.5 * self.perceived_effectiveness_weir + 0.5 * self.self_efficacy), 1) 
            self.coping_appraisal_channel = min((0.5 * self.perceived_effectiveness_channel + 0.5 * self.self_efficacy), 1)

            self.intention_to_adapt_weir = self.alpha * self.threat_appraisal + self.beta * self.coping_appraisal_weir
            self.intention_to_adapt_channel = self.alpha * self.threat_appraisal + self.beta * self.coping_appraisal_channel
