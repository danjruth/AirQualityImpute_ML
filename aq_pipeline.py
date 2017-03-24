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
if ~('all_data_c' in locals()):
    all_data_c = pd.read_csv(aq.all_data_path,usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean','Parameter Code','Latitude','Longitude'])
    all_data_c = all_data_c.rename(columns={'Site Num':'Site Number'})

all_data = all_data_c.copy()

latlon = (34.1522, -118.2437)
r_max = 100

start_date = '2011-01-01'
end_date = '2013-06-30'

# this will store the metadata for each station that'll be used
stations = aq.identify_nearby_stations(latlon,r_max,all_data.copy())
stations = aq.addon_stationid(stations)
stations = aq.remove_dup_stations(stations)
print(stations)

stations = aq.create_station_weights(stations)
print(stations)

# plot these stations on a map
#aq.plot_station_locs(stations)

orig = pd.DataFrame(columns=stations.index.copy())

# for each nearby station, fill in missing data
composite_data = pd.DataFrame()
#for i in range(0,len(stations)):
for station in stations.index:

    station_obj = None
    
    station_obj =aq.aq_station(station)
    station_obj.latlon = (stations.loc[station,'Latitude'],stations.loc[station,'Longitude'])
    station_obj.start_date = start_date
    station_obj.end_date = end_date
    station_obj.get_station_data(r_max,all_data.copy())
    orig.loc[:,station] = station_obj.this_station.copy()
    station_obj.create_model()
    station_obj.run_model()
    
    composite_data.loc[:,station] = station_obj.composite_data.rename(station).copy()
    
data = aq.spatial_interp(composite_data,stations)   
aq.final_big_plot(data,orig,composite_data,stations)

aq.matrix_val_plot(orig)
aq.matrix_val_plot(composite_data)

