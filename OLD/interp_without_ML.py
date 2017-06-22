# -*- coding: utf-8 -*-
"""
Created on Wed May 24 20:19:14 2017

@author: danjr
"""

import AQ_ML as aq
import pandas as pd
import matplotlib.pyplot as plt

meta = pd.read_csv('C:\Users\danjr\Documents\ML\Air Quality\\aq_comparison_metadata.csv',index_col='Name')

start_date = '2012-01-01'
end_date = '2015-12-31'

r_max_interp = 300

all_data_all = aq.extract_raw_data(start_date,end_date,param_code=81102)

res = {}
for ix in meta.index:
    latlon = (meta.loc[ix,'lat'],meta.loc[ix,'long'])
    
    start_date = pd.to_datetime(meta.loc[ix,'start date'])
    end_date = start_date + pd.Timedelta('366D')
    
    nearby_stations = aq.identify_nearby_stations(latlon,r_max_interp,all_data_all.copy(),start_date,end_date,ignore_closest=False)
    nearby_stations = aq.addon_stationid(nearby_stations)
    nearby_stations = aq.remove_dup_stations(nearby_stations,ignore_closest=False)    
    
    nearby_data = aq.extract_nearby_values(nearby_stations,all_data_all,start_date,end_date)
    
    res[ix] = aq.spatial_interp_variable_weights(nearby_data,nearby_stations,max_stations=20)
    
    plt.figure()
    plt.plot(res[ix])
    plt.show()
    plt.pause(.1)
    
    
for ix in list(res.keys()):
    res[ix].to_csv('C:\Users\danjr\Documents\ML\Air Quality\whitepaper_data\\'+ix+'.csv')