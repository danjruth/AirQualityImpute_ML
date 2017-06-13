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

r_max_interp = 150 # how far from latlon of interest should it look for stations?
r_max_ML = 250 # for each station it finds, how far should it look aroud it in imputing the missing values?

predict_dates = ['2014-01-01','2014-12-31']
period_dates = {4:['2011-01-01','2014-12-31'],3:['2012-01-01','2014-12-31'],2:['2013-01-01','2014-12-31'],1:['2014-01-01','2014-12-31']}


rows_to_test = aq.addon_stationid(aq.extract_raw_data(predict_dates[0],predict_dates[1],param_code=81102).sample(200)).set_index('station_ids')

res = pd.DataFrame(index=rows_to_test.index,columns=list(period_dates.keys()))

for station in rows_to_test.index:
    lat = rows_to_test.loc[station,'Latitude']
    lon = rows_to_test.loc[station,'Longitude']

    for data_ranges in list(period_dates.keys()):
        
        try:
        
            dates = period_dates[data_ranges]
            
            all_data_all = aq.extract_raw_data(dates[0],dates[1],param_code=81102)
            
            pm25_data = aq.extract_raw_data(dates[0],dates[1],param_code=88101)
            CO_data = aq.extract_raw_data(dates[0],dates[1],param_code=42101)
            other_data_all = pd.concat([pm25_data,CO_data])
            other_data_all = other_data_all.set_index(pd.Series(data=range(len(other_data_all)))) # reindex to get rid of duplicate indices (index here is not significant)
                
            station_obj = aq.aq_station(station,ignoring=None)
            station_obj.latlon = (lat, lon)
            station_obj.start_date = predict_dates[0]
            station_obj.end_date = predict_dates[1]
            
            # extract data from nearby stations in the EPA database
            station_obj.get_station_data(r_max_ML,all_data_all.copy(),other_data_all.copy())
            
            # create and run a model to fill in missing data
            station_obj.create_model()
            station_obj.run_model()
        
            res.loc[station,data_ranges] = max(station_obj.model_r2)
            
        except:
            res.loc[station,data_ranges] = None
            plt.pause(1) # give time to actually cancel the execution
            
    res.to_pickle(r'C:\Users\danjr\Documents\ML\Air Quality\time_periods_testb.pkl')
    plt.close('all')
            
fig = plt.figure()
ax = fig.add_subplot(111)

for station in res.index:
    
    x = []
    y = []
    
    for i in res.columns:
        x.append(i)
        y.append(res.loc[station,i])
        
    ax.plot(x,y,alpha=0.2)