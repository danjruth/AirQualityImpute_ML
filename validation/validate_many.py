# -*- coding: utf-8 -*-
"""
Created on Fri May 05 08:50:32 2017

@author: druth
"""

import AQ_ML as aq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

'''
This script is used to compare the estimation methods (with and without 
imputation) to known data.

Coordinates of air quality stations reporting PM10 concentrations daily are 
stored in test_locations.csv. This script can pick out a random sample of those
stations, remove their readings from  the dataset, perform the estimation, then
compare the results to the known values.

The script validation_plots.py can be run after this to plot the metrics
characterizing each approach for each station tested against.
'''

### ---- USER INPUTS ---- ###

r_max_interp = 150 # how far from latlon of interest should it look for stations?
r_max_ML = 250 # for each station it finds, how far should it look aroud it in imputing the missing values?

start_date = '2012-01-01'
end_date = '2014-12-31'


all_data_all = aq.extract_raw_data(start_date,end_date,param_code=81102)

pm25_data = aq.extract_raw_data(start_date,end_date,param_code=88101)
CO_data = aq.extract_raw_data(start_date,end_date,param_code=42101)
other_data_all = pd.concat([pm25_data,CO_data])
#other_data = pd.concat([pm25_data])
other_data_all = other_data_all.set_index(pd.Series(data=range(len(other_data_all)))) # reindex to get rid of duplicate indices (index here is not significant)


latlons = pd.read_csv('C:\Users\danjr\Documents\ML\Air Quality\Code\\validation\\test_locations.csv')
#latlons = pd.read_csv(r'C:\Users\druth\Documents\AirQualityImpute_ML\validation\test_locations.csv')
ix = 42
results = pd.DataFrame(columns=latlons.columns)
results.loc[ix,:] = latlons.loc[ix,:]
results_dict = {}

for ix in results.index:
    
    latlon = (results.loc[ix]['Latitude'],results.loc[ix]['Longitude'])
    
    all_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,all_data_all.copy(),start_date,end_date,ignore_closest=False)
    other_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,other_data_all.copy(),start_date,end_date,ignore_closest=False)
    all_data = all_data.sort_values('Date Local')
    other_data = other_data.sort_values('Date Local')
    
    start_date = '2014-01-01'
    end_date = '2014-12-31'
    composite_data, orig, stations, station_obj_list, target_data = aq.create_composite_dataset(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data.copy(),other_data.copy(),ignore_closest=True)
    data = aq.predict_aq_vals(composite_data,stations)
    
    nearby_stations = aq.identify_nearby_stations(latlon,r_max_interp,all_data.copy(),start_date,end_date)
    nearby_stations = aq.addon_stationid(nearby_stations)
    nearby_stations = aq.remove_dup_stations(nearby_stations,ignore_closest=True)    
    nearby_data = aq.extract_nearby_values(nearby_stations,all_data.copy(),start_date,end_date)
    results_noML = aq.spatial_interp_variable_weights(nearby_data,nearby_stations,max_stations=10)
    
    target_data = target_data[(target_data.index>=start_date)&(target_data.index<=end_date)]
    results_dict[ix] = (data, target_data, results_noML, station_obj_list, composite_data, orig)
    
    compare_df = pd.DataFrame()
    compare_df['predicted'] = data.copy()
    compare_df['predicted_noML'] = results_noML.copy()
    compare_df['target'] = target_data.copy()
    compare_df_all = compare_df.copy()
    # only keep rows for which there is target data to compare against
    compare_df = compare_df[np.isfinite(compare_df['target'])]
    #compare_df = compare_df.fillna(0) # think more about this
    compare_df = compare_df.fillna(method='ffill').fillna(method='bfill')
    
    ## Compute/print some metrics
    
    from sklearn.metrics import r2_score, mean_absolute_error
    
    # window for the rolling mean calculations.
    # rolling mean is useful since values reported for soiling periods are averages
    # over at least 14 days.
    win = 14
    
    print('----- METRICS -----')
    
    r2 = r2_score(compare_df['predicted'],compare_df['target'])
    r2_noML = r2_score(compare_df['predicted_noML'],compare_df['target'])
    print('R squareds (with, without ML) are:')
    print(r2,r2_noML)
     
    r2_roll = r2_score(compare_df['predicted'].rolling(window=win,min_periods=0).mean(),compare_df['target'].rolling(window=win,min_periods=0).mean())
    r2_roll_noML = r2_score(compare_df['predicted_noML'].rolling(window=win,min_periods=0).mean(),compare_df['target'].rolling(window=win,min_periods=0).mean())
    print('Rolling r squareds (with, without ML) are:')
    print(r2_roll,r2_roll_noML)
    
    corr = compare_df['target'].corr(compare_df['predicted'])
    corr_noML = compare_df['target'].corr(compare_df['predicted_noML'])
    print('Pearson correlations (with, without ML) are:')
    print(corr,corr_noML)
    
    corr_roll = compare_df['target'].rolling(window=win,min_periods=0).mean().corr(compare_df['predicted'].rolling(window=win,min_periods=0).mean())
    corr_roll_noML = compare_df['target'].rolling(window=win,min_periods=0).mean().corr(compare_df['predicted_noML'].rolling(window=win,min_periods=0).mean())
    print('Rolling Pearson correlations (with, without ML) are:')
    print(corr_roll,corr_roll_noML)
    
    mae = mean_absolute_error(compare_df['predicted'],compare_df['target'])
    mae_noML = mean_absolute_error(compare_df['predicted_noML'],compare_df['target'])
    print('Mean abs. errors (with, without ML) are:')
    print(mae,mae_noML)
    
    mae_roll = mean_absolute_error(compare_df['predicted'].rolling(window=win,min_periods=0).mean(),compare_df['target'].rolling(window=win,min_periods=0).mean())
    mae_roll_noML = mean_absolute_error(compare_df['predicted_noML'].rolling(window=win,min_periods=0).mean(),compare_df['target'].rolling(window=win,min_periods=0).mean())
    print('Rolling mean abs. errors (with, without ML) are:')
    print(mae_roll,mae_roll_noML)
    
    results.loc[ix,'r2'] = r2
    results.loc[ix,'r2_noML'] = r2_noML
    results.loc[ix,'r2_roll'] = r2_roll
    results.loc[ix,'r2_roll_noML'] = r2_roll_noML
    
    results.loc[ix,'corr'] = corr
    results.loc[ix,'corr_noML'] = corr_noML
    results.loc[ix,'corr_roll'] = corr_roll
    results.loc[ix,'corr_roll_noML'] = corr_roll_noML
    
    results.loc[ix,'mae'] = mae
    results.loc[ix,'mae_noML'] = mae_noML
    results.loc[ix,'mae_roll'] = mae_roll
    results.loc[ix,'mae_roll_noML'] = mae_roll_noML
    
    # plot the results against target data
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(211)
    ax1.plot(results_noML,'.-',color='gray',label='predicted, without imputation')
    ax1.plot(data,'.-',color='k',label='predicted, with imputation')
    ax1.plot(target_data,'.-',color='green',label='target')
    ax1.legend()
    ax1.set_title('Instantaneous')
    plt.show()
    
    # plot the results against target data
    ax2 = fig.add_subplot(212,sharex=ax1,sharey=ax1)
    ax2.plot(results_noML.rolling(window=win,min_periods=0).mean(),'.-',color='gray',label='predicted, without imputation')
    ax2.plot(data.rolling(window=win,min_periods=0).mean(),'.-',color='k',label='predicted, with imputation')
    ax2.plot(target_data.rolling(window=win,min_periods=0).mean(),'.-',color='green',label='target')
    ax2.set_title('Rolling')
    ax2.legend()
    ax1.set_ylabel('PM10 Concentration')
    ax2.set_ylabel('PM10 Concentration')
    plt.show()
    
    print(results)