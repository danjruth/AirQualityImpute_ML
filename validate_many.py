# -*- coding: utf-8 -*-
"""
Created on Fri May 05 08:50:32 2017

@author: druth
"""

import AQ_ML as aq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### ---- USER INPUTS ---- ###

start_date = '2013-01-01'
end_date = '2014-12-31'

r_max_interp = 100 # how far from latlon of interest should it look for stations?
r_max_ML = 250 # for each station it finds, how far should it look aroud it in imputing the missing values?

all_data_all = aq.extract_raw_data(start_date,end_date,param_code=81102)

pm25_data = aq.extract_raw_data(start_date,end_date,param_code=88101)
#ozone_data = aq.extract_raw_data(start_date,end_date,param_code=44201)
CO_data = aq.extract_raw_data(start_date,end_date,param_code=42101)
other_data_all = pd.concat([pm25_data,CO_data])
#other_data = pd.concat([pm25_data])
other_data_all = other_data_all.set_index(pd.Series(data=range(len(other_data_all)))) # reindex to get rid of duplicate indices (index here is not significant)

latlons = pd.read_csv('C:\Users\druth\Documents\AirQualityImpute_ML\\test_locations.csv')
results = latlons.sample(20)
results_dict = {}

for ix in results.index:
    
    try:
        latlon = (results.loc[ix]['Latitude'],results.loc[ix]['Longitude'])
        
        all_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,all_data_all.copy(),start_date,end_date,ignore_closest=False)
        other_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,other_data_all.copy(),start_date,end_date,ignore_closest=False)
        all_data = all_data.sort_values('Date Local')
        other_data = other_data.sort_values('Date Local')
        
        data, target_data, results_noML, station_obj_list, composite_data, orig, all_stations = aq.predict_aq_vals(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data,other_data,ignore_closest=True,return_lots=True)
    
        results_dict[ix] = (data, target_data, results_noML, station_obj_list, composite_data, orig, all_stations)
        
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
        
        print(results)
        
    except:
        print('DID NOT WORK!')
        print(results)