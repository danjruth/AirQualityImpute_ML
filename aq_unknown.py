# -*- coding: utf-8 -*-
"""

Runs the algorithm implemented in AQ_ML.py

Created on Sun Mar 12 19:07:18 2017

@author: danjr
"""

import AQ_ML as aq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### ---- USER INPUTS ---- ###

start_date = '2011-01-01'
end_date = '2015-12-31'

latlon = (34.79,-118.51)
r_max_interp = 250 # how far from latlon of interest should it look for stations?
r_max_ML = 250 # for each station it finds, how far should it look aroud it in imputing the missing values?

### ---- END USER INPUTS ---- ###

# get the raw daa

all_data = aq.extract_raw_data(start_date,end_date,param_code=81102)

pm25_data = aq.extract_raw_data(start_date,end_date,param_code=88101)
ozone_data = aq.extract_raw_data(start_date,end_date,param_code=44201)
CO_data = aq.extract_raw_data(start_date,end_date,param_code=42101)
other_data = pd.concat([pm25_data,CO_data,ozone_data])
other_data = other_data.set_index(pd.Series(data=range(len(other_data)))) # reindex to get rid of duplicate indices (index here is not significant)

# filter out stations that are definitley too far away to be of any use.
# we know nothing farther than r_max_interp+r_max_ML will be used.
# not necessary, but it should save time.
all_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,all_data,start_date,end_date,ignore_closest=False)
other_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,other_data,start_date,end_date,ignore_closest=False)
all_data = all_data.sort_values('Date Local')
other_data = other_data.sort_values('Date Local')


data, station_obj_list, composite_data, orig, stations = aq.predict_aq_vals(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data,other_data,ignore_closest=False,return_lots=True)
data_noML = aq.spatial_interp_variable_weights(orig,stations,max_stations=5)

win = 14
fig = plt.figure(figsize=(6,8))

ax1 = fig.add_subplot(211)
ax1.plot(data_noML,color='gray')
ax1.plot(data,color='k')

ax2 = fig.add_subplot(212,sharex=ax1,sharey=ax1)
ax2.plot(data_noML.rolling(window=win,min_periods=0).mean(),color='gray')
ax2.plot(data.rolling(window=win,min_periods=0).mean(),color='k')