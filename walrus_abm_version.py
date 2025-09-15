import numpy as np
import pandas as pd
from datetime import datetime
from scipy.optimize import root_scalar
from numba import njit 
# from WALRUS_code_HydroFarm import p_num


def WALRUS_date_to_seconds(d):
    """
    Converts yyyymmddhh or yyyymmdd to seconds since 1970-01-01.
    d = date/time combination in format yyyymmdd or yyyymmddhh
    returns number of seconds since 1970-01-01
    """
    if len(str(d)) == 12:
        seconds = (datetime.strptime(str(d), '%Y%m%d%H%M') - datetime(1970, 1, 1)).total_seconds()
    elif len(str(d)) == 10:
        seconds = (datetime.strptime(str(d), '%Y%m%d%H') - datetime(1970, 1, 1)).total_seconds()
    elif len(str(d)) == 8:
        seconds = (datetime.strptime(str(d), '%Y%m%d') - datetime(1970, 1, 1)).total_seconds()
    else:
        raise ValueError('date has not format YYYYmmdd or YYYYmmddHH')

    return seconds

def WALRUS_select_dates(input, start, end):
    """
    This function cuts out the part of the dataset between the start- and end date.
    input =  dataframe with date in YYYYmmddhh or YYYmmdd format and columns with P, ETpot, Q (at least one value), fXG (optional), fXS (optional), GWL (optional)
    start =  start  of simulation (in YYYYmmddhh or YYYmmdd format)
    end   =  end of the simulation (in YYYYmmddhh or YYYmmdd format)
    returns a shorter dataframe of the input data 
   
    Note that the format for start and end differs for daily and hourly data:
    for hourly data:  format: yyyymmddhh
    for daily data:   format: yyyymmdd
    """
    global forc
    forc = input[(input["date"] >= start) & (input["date"] <= end)]
    return forc

def WALRUS_preprocessing(f, dt, timestamp="start"):
    pd.set_option('mode.chained_assignment', None)
    # Check for missing variables and if nescessary create column for missing variable
    if 'P' not in f.columns:
        raise FileNotFoundError("Please supply P in forcing data frame")
    if 'ETpot' not in f.columns:
        raise FileNotFoundError("Please supply ETpot in forcing data frame")
    if 'Q' not in f.columns:
        f['Q'] = np.zeros(f.shape[0])
    if 'fXG' not in f.columns:
        f['fXG'] = np.zeros(f.shape[0])
    if 'fXS' not in f.columns:
        f['fXS'] = np.zeros(f.shape[0])
    if 'hSmin' not in f.columns:
        f['hSmin'] = np.zeros(f.shape[0])
    if 'dG' not in f.columns:
        f['dG'] = np.zeros(f.shape[0])
    if 'warm' not in f.columns:
        f['warm'] = 0

    # Check for missing values in each column and fill them accordingly
    if f['P'].isna().sum() > 0:
        f.fillna({'P': 0}, inplace=True)
        print(f"Note: missing P set to zero ({f['P'].isna().sum()} cases)")

    for col in f.columns:       
        if f[col].isna().sum() > 0:
            # print(f"Note: missing {col} interpolated ({f[col].isna().sum()} cases)")
            f[col] = f[col].interpolate(method='linear', limit_direction='both')  # Linear interpolation

    # write date as number of seconds since 1970
    if 'date' not in f.columns:
        raise FileNotFoundError("date is not in dataframe, provide column with dates in format yyyymmdd or yyyymmddhh")
    else:
        con_list =[] 
        for i in f['date']:
            converted = WALRUS_date_to_seconds(i)
            con_list.append(converted)
    
    f['date'] = con_list

    # if timestamps belong to the end of measurement period (for fluxes), move the dates forward, 
    # such that the timestamps belong to the start of the measurement period.
    # Move states (dG, hSmin) 
    if timestamp == "end":
        f['date'] = f['date'].shift(periods=1)
        f.loc[0, "date"] = f.loc[1, "date"] - (f.loc[2, "date"] - f.loc[1, "date"])
        f['dG'] = f['dG'].shift(periods=1)
        f['hSmin'] = f['hSmin'].shift(periods=1)

    dates = f['date']
    d_end = f['date'].shift(periods=-1)
    d_end.iloc[-1] = d_end.iloc[-2] + (d_end.iloc[-2] - d_end.iloc[-3])

    global forcing_date
    forcing_date = pd.concat([dates, d_end], axis= 1)
    forcing_date.columns = ["date", 'date_end']

       # Calculate number of output dates
    nr = int(np.floor((len(forcing_date) - 1) / dt))
    # Create index array
    idx = np.concatenate(([0], np.arange(1, nr * dt + 1, dt)))
    # Select output dates
    global output_date
    output_date = forcing_date.iloc[idx]

    output_date.index = range(len(output_date.index))
    return output_date

# Soil characteristics
soil_char = {
    1: {"b": 4.38, "psi_ae": 90, "theta_s": 0.410},
    "sand": {"b": 4.05, "psi_ae": 121, "theta_s": 0.395},
    "loamy_sand": {"b": 4.38, "psi_ae": 90, "theta_s": 0.410},
    "sandy_loam": {"b": 4.90, "psi_ae": 218, "theta_s": 0.435},
    "silt_loam": {"b": 5.30, "psi_ae": 786, "theta_s": 0.485},
    "loam": {"b": 5.39, "psi_ae": 478, "theta_s": 0.451},
    "sandy_clay_loam": {"b": 7.12, "psi_ae": 299, "theta_s": 0.420},
    "silt_clay_loam": {"b": 7.75, "psi_ae": 356, "theta_s": 0.477},
    "clay_loam": {"b": 8.52, "psi_ae": 630, "theta_s": 0.476},
    "sandy_clay": {"b": 10.40, "psi_ae": 153, "theta_s": 0.426},
    "silty_clay": {"b": 10.40, "psi_ae": 490, "theta_s": 0.492},
    "clay": {"b": 11.40, "psi_ae": 405, "theta_s": 0.482},
    "cal_H": {"b": 2.63, "psi_ae": 90, "theta_s": 0.418},
    "cal_C": {"b": 16.77, "psi_ae": 9, "theta_s": 0.639}
}

# Default values parameters of default parameterizations
p_default = {
    "zeta1": 0.02,
    "zeta2": 400,
    "expS": 1.5
}
p_default_numba = np.array(list(p_default.values()), dtype=np.float64)
# Default values numeric parameters
p_num = {
    "min_h": 0.001,
    "max_P_step": 10,
    "max_dQ_step": 0.1,
    "max_h_change": 10,
    "min_timestep": 60
}
p_num_numba = np.array(list(p_num.values()), dtype=np.float64)

################# NUMBA FUNCTIONS ################
@njit
def func_W_dV_numba(x, pars):
    return (np.cos(np.maximum(np.minimum(x, pars[0]), 0) * np.pi / pars[0]) / 2 + 0.5)

def func_W_dV_normal(x, pars):
    return (np.cos(np.maximum(np.minimum(x, pars['cW']), 0) * np.pi / pars['cW']) / 2 + 0.5)

@njit
def func_dVeq_dG_numba(x, pars):
    if x > pars[10]:
        return (x - pars[10] / (1 - pars[9]) - x * (x / pars[10]) ** (-1 / pars[9]) +
                pars[10] / (1 - pars[9]) * (x / pars[10]) ** (1 - 1 / pars[9])) * pars[11]
    elif x < 0:
        return x
    else:
        return 0

def func_dVeq_dG_normal(x, pars):
    if x > pars['psi_ae']:
        return (x - pars['psi_ae'] / (1 - pars['b']) - x * (x / pars['psi_ae']) ** (-1 / pars['b']) +
                pars['psi_ae'] / (1 - pars['b']) * (x / pars['psi_ae']) ** (1 - 1 / pars['b'])) * pars['theta_s']
    elif x < 0:
        return x
    else:
        return 0

@njit
def func_beta_dV_numba(x):
    return ((1 - np.exp(-p_default_numba[0] * (p_default_numba[1] - x))) /(1 + np.exp(-p_default_numba[0] * (p_default_numba[1] - x))) / 2 + 0.5)

def func_beta_dV_normal(x):
    return ((1 - np.exp(-p_default["zeta1"] * (p_default["zeta2"] - x))) /
            (1 + np.exp(-p_default["zeta1"] * (p_default["zeta2"] - x))) / 2 + 0.5)

@njit
def func_Q_hS_numba(hs, pars, hSmin):  
    if hs <= hSmin:
        return 0
    elif hs <= pars[5]:
        return pars[8] * ((hs - hSmin) / (pars[5] - hSmin)) ** 1.5
    else:
        return pars[8] + (pars[8] * ((hs - hSmin) / (pars[5] - hSmin)) ** 1.5)

def func_Q_hS_normal(hs, pars, hSmin):  
    if hs <= hSmin:
        return 0
    elif hs <= pars['cD']:
        # return pars['cS'] / (pars['cD'] - hSmin) ** p_default['expS'] * (hs - hSmin) ** p_default['expS']
        return pars['cS'] * ((hs - hSmin) / (pars['cD'] - hSmin)) ** p_default['expS']
    else:
        return pars['cS'] + (pars['cS'] * ((hs - hSmin) / (pars['cD'] - hSmin)) ** p_default['expS'])

# Helper to convert array to dict
def WALRUS_array_to_dict(arr, order):
    return {k: arr[i] for i, k in enumerate(order)}

def WALRUS_dict_to_array(d, order):
    return np.array([d[k] for k in order], dtype=np.float64)


@njit
def WALRUS_step_numba(pars, i, P_t, ETpot_t, fXG_t, fXS_t, hSmin, dt_in):
    '''
    One of the main WALRUS-functions. Computes states and fluxes for one time step
    pars = the parameter set used for the run
    i = initital states for this time step
    t1 = start of the time step (in seconds since 1970-01-01)
    t2 = end of the time step (in seconds since 1970-01-01)
    return a vector with the model output for one time step.
    '''

    # STEPSIZE
    dt = dt_in  # in hours   # compute dt (in hours because parameters are in hours)
    dt_ok = 1            # stepsize small enough as default

    # FLUXES
    PQ = P_t * i[9] * pars[12]
    PV = P_t * (1 - i[9]) * pars[12]
    PS = P_t * pars[6]
    ETV = ETpot_t * func_beta_dV_numba(i[4]) * pars[12]
    ETS = ETpot_t * pars[6]
    if i[8] < p_num_numba[0] * 1000:
        ETS = 0                             # no ET from empty channel
    ETact = ETV + ETS
    fQS = i[7] / pars[3] * dt
    fGS = (pars[5] - i[6] - i[8]) * max((pars[5] - i[6]), i[8]) / (pars[2])* dt
    Q = func_Q_hS_numba(i[8], pars, hSmin) * dt
    
    ### STATES (at the end of this time step / start of next time step) [mm])
    # note that fluxes are already for the whole time step (multiplied with dt)
    dV = i[4] - (fXG_t + PV - ETV - fGS) / pars[12]
    hQ = i[7] + (PQ - fQS) / pars[12]
    hS = i[8] + (fXS_t + PS - ETS + fGS + fQS - Q) / pars[6]
    dG = i[6] + (i[4] - i[5]) / pars[1] * dt

    # SPECIAL CASE: LARGE-SCALE PONDING AND FLOODING
    if dV < 0 or hS > pars[5]:
        if dV < 0 and hS <= pars[5]:                             # if ponding and no flooding
            hS = hS + (-dV) * pars[12] / pars[6]               # all ponds to surface water
            dV = 0                                                  # soil moisture deficit to surface
        if dV >= 0 and hS > pars[5]:                             # if no ponding and flooding
            dV = dV - (hS - pars[5]) * pars[6] / pars[12]   # all floods into soil
            hS = pars[5]                                         # channel bankfull
        if dV <= 0 and hS >= pars[5]:                            # if ponding and flooding
            dV = dV * pars[12] - (hS - pars[5]) * pars[6]   # compute total excess water
            hS = pars[5] - dV
        if dV < 0:
            dG = dV                                                 # if ponding, groundwater to pond level
    # TEST IF STEP SIZE IS SMALL ENOUGH
    if hS < -p_num_numba[0]:            # if hS below channel bottom
        dt_ok = 0
        hS = p_num_numba[0] * 100
    elif hQ < -p_num_numba[0]:          # if hQ below bottom Q-res.
        dt_ok = 0
        hQ = p_num_numba[0]
    elif P_t > p_num_numba[1]:     # if too much rainfall added
        dt_ok = 0
    elif abs(i[1] - Q) > p_num_numba[2]:    # if change in Q too big
        dt_ok = 0
    elif abs(i[8] - hS) > p_num_numba[3]: # if change in hS too big
        dt_ok = 0
    elif abs(i[6] - dG) > p_num_numba[3]: # if change in dG too big
        dt_ok = 0

    # OUTPUT
    # compute dependent variables (at end of time step)
    W = func_W_dV_numba(dV, pars)
    dVeq = func_dVeq_dG_numba(dG, pars)

    # return a list with the model output for one time step
    return [ETact, Q, fGS, fQS, dV, dVeq, dG, hQ, hS, W, dt_ok, hSmin, fXS_t, fXG_t]

def WALRUS_prepare_loop(pars, output_date, hydro_stat, func_Qobs, func_hSmin):
    o = hydro_stat
    # Look up soil type parameters
    soil_type = pars["st"]
    soil_char_subset = soil_char[soil_type]
    pars["b"] = soil_char_subset["b"]
    pars["psi_ae"] = soil_char_subset["psi_ae"]
    pars["theta_s"] = soil_char_subset["theta_s"]
    pars["aG"] = 1 - pars["aS"]

    ######################
    ### INITIAL CONDITIONS 
    ######################
    # Q[1] is necessary for stepsize-check (if dQ too large)
    if "Q0" in pars:
        o.loc[0, "Q"] = pars["Q0"]
    else:
        #o.loc[0, "Q"] = func_Qobs(output_date.loc[0, 'date'])
        o.loc[0, "Q"] = func_Qobs(output_date.loc[1, 'date']) / (output_date.loc[1, 'date'] - output_date.loc[0, 'date']) * 3600
    
    # hS from first Q measurement and Qh-relation
    if "hS0" in pars:
        o.loc[0, "hS"] = pars["hS0"]
    else:
        hS_root = root_scalar(lambda x: func_Q_hS_normal(x, pars, func_hSmin(output_date.loc[0, 'date'])) - o.loc[0, "Q"], 
                              bracket=[0, pars['cD']])
        o.loc[0, 'hS'] = hS_root.root

    # dG and hQ
    if "dG0" in pars:                                           # if dG0 provided 
        o.loc[0, "dG"] = pars["dG0"]
        if "hQ0" in pars:                                       # if hQ0 also provided 
            o.loc[0, "hQ"] = pars["hQ0"]
        else:                                                   # if hQ0 not provided 
            if pars["cD"] - o.loc[0, "dG"] < o.loc[0, "hS"]:    # if groundwater below surface water level
                o.loc[0, "hQ"] = o.loc[0, "Q"] * pars["cQ"]     # all Q from quickflow
            else:                                               # if groundwater above surface water level
                o.loc[0, "hQ"] = max(0, (o.loc[0, "Q"] - (pars["cD"] - o.loc[0, "dG"] - o.loc[0, "hS"]) *
                                        (pars["cD"] - o.loc[0, "dG"]) / pars["cG"]) * pars["cQ"])
      
    elif "dG0" not in pars and "hQ0" in pars:
        o.loc[0, "hQ"] = pars["hQ0"]
        dG_root = root_scalar(lambda x: ((pars["cD"] - x - o.loc[0, "hS"]) * (pars["cD"] - x) / pars["cG"] - o.loc[0, "Q"] * pars["Gfrac"]),
                                 bracket=[0, pars["cD"] - o.loc[0, "hS"]])
        o.loc[0, "dG"] = dG_root.root
    else:                                       # if dG0 not provided 
        if "Gfrac" not in pars:                 
            pars["Gfrac"] = 1                   # if Gfrac also not provided, make Gfrac 1
        # if fGS not possible with current hS and cG, make Gfrac smaller
        while (pars["cD"] - o.loc[0, "hS"]) * pars["cD"] / pars["cG"] < pars["Gfrac"] * o.loc[0, "Q"]:
            pars["Gfrac"] = pars["Gfrac"] / 2
        # compute dG leading to the right fGS
        dG_root = root_scalar(lambda x: ((pars["cD"] - x - o.loc[0, "hS"]) * (pars["cD"] - x) / pars["cG"] - o.loc[0, "Q"] * pars["Gfrac"]),
                                 bracket=[0, pars["cD"] - o.loc[0, "hS"]])  
        o.loc[0, "dG"] = dG_root.root       
        o.loc[0, "hQ"] = o.loc[0, "Q"] * (1 - pars["Gfrac"]) * pars["cQ"]

    # dVeq
    o.loc[0, "dVeq"] = func_dVeq_dG_normal(o.loc[0, "dG"], pars)

    # dV
    if "dV0" in pars:
        o.loc[0, "dV"] = pars["dV0"]
    else:
        o.loc[0, "dV"] = o.loc[0, "dVeq"]
    
    # W
    o.loc[0, "W"] = func_W_dV_normal(o.loc[0, "dV"], pars)

    # Prepare for-loop
    o_step = o.iloc[0]
    i = o.iloc[0]
    return i

def WALRUS_loop_numba(pars, date, hydro_stat,input_walrus_init, func_P, func_ETpot, func_fXG, func_fXS):
    for t in range(1, len(date)):
        start_step = date.loc[t-1, 'date']   # start at begin of output step
        end_step = date.loc[t, 'date']       # first try whole output step
        sums_step = np.zeros(4)
        hSmin = 0
    
        SOIL_PARS_ORDER = ["cW", "cV", "cG", "cQ", "dG0", "cD", "aS", "st", "cS", "b", "psi_ae", "theta_s", "aG"]
        HYDRO_STAT_ORDER = ["ETact", "Q", "fGS", "fQS", "dV", "dVeq", "dG", "hQ", "hS", "W", "dt_ok", "hSmin", "fXS", "fXG"]

        soil_pars_arr = np.array([pars[key] for key in SOIL_PARS_ORDER], dtype=np.float64)
        input_walrus_arr = np.array([input_walrus_init[key] for key in HYDRO_STAT_ORDER], dtype=np.float64)

        while start_step < (date.loc[t, 'date']  - p_num["min_timestep"]):
            P_t = func_P(end_step) - func_P(start_step)
            ETpot_t = func_ETpot(end_step) - func_ETpot(start_step)
            fXG_t = func_fXG(end_step) - func_fXG(start_step)
            fXS_t = func_fXS(end_step) - func_fXS(start_step)
            dt = (end_step - start_step) / 3600  # in hours
            
            step = WALRUS_step_numba(soil_pars_arr, input_walrus_arr, P_t, ETpot_t, fXG_t, fXS_t, hSmin, dt)

            step_dict = WALRUS_array_to_dict(step, HYDRO_STAT_ORDER)

            if step_dict["dt_ok"] == 0 and (int(end_step) - int(start_step)) > p_num["min_timestep"]:
                end_step = (int(start_step) + int(end_step)) / 2
            else:
                start_step = end_step
                end_step = date.loc[t, 'date']
                sums_step += step[0:4]
                hydro_stat.iloc[t] = pd.Series(step_dict)
                input_walrus_init = step_dict
                input_walrus_arr = WALRUS_dict_to_array(input_walrus_init, HYDRO_STAT_ORDER)
    
        hydro_stat.iloc[t] = pd.Series(step_dict)
        hydro_stat.iloc[t, 0:4] = sums_step
    return hydro_stat
