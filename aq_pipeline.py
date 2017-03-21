# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:07:18 2017

@author: danjr
"""

import AQ_ML as aq
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np

#station_df = pd.read_csv(aq.station_df_path)
#all_data = pd.read_csv(aq.all_data_path,usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean','Parameter Code','Latitude','Longitude'])
#all_data = all_data.rename(columns={'Site Num':'Site Number'})


latlon = (33.424564, -111.928001) # ASU
r_max = 100

start_date = '2012-03-01'
end_date = '2013-01-01'

#station_data = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='1D').date)

'''
stations = aq.identify_nearby_stations(latlon,r_max,all_data)
stations = aq.addon_stationid(stations)
stations = aq.remove_dup_stations(stations)
'''


#nearby_data = aq.extract_nearby_values(stations,all_data,start_date,end_date)

# split up the stations in to good stations (enough data) and bad ones (to be imputed)
gs,bs = aq.split_fill_unfill_stations(nearby_data)

# replace missing data in predictors (won't be too many of these)
gs = aq.fill_missing_predictors(gs)

# show the good and bad stations
fig = plt.figure()
ax_g = fig.add_subplot(2,1,1)
ax_g.matshow(gs.transpose())
ax_b = fig.add_subplot(2,1,2)
ax_b.matshow(bs.transpose())

# fill in missing data in each "bad column"
for column in bs:
    col_vals = bs[column]
    aq.create_model_for_site(gs,col_vals)
