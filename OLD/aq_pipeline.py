# -*- coding: utf-8 -*-
"""

Runs the algorithm implemented in AQ_ML.py

Created on Sun Mar 12 19:07:18 2017

@author: danjr
"""

import AQ_ML as aq
import pandas as pd
import matplotlib.pyplot as plt

'''
Run this script to estimate the PM10 concentration at a point, both with and 
without the machine learning imputation procedure applied.
'''

'''
### ---- USER INPUTS ---- ###
'''

# range in which to estimate air quality data
start_date = '2013-01-01'
end_date = '2014-12-31'

# location at which to estimate air quality data
latlon = (39,	-122)

# range for data to be used in creating models to fill in missing station data
start_date_data = '2012-01-01'
end_date_data = '2015-12-31'

# distance parameters
r_max_interp = 200 # how far from latlon of interest should it look for stations?
r_max_ML = 200 # for each station it finds, how far should it look aroud it in imputing the missing values?

'''
### ---- END USER INPUTS ---- ###
'''

'''
Prepare the data.
'''

# extract the air quality data of the species to be predicted
all_data = aq.extract_raw_data(start_date_data,end_date_data,param_code=81102)

# extract the auxillary air quality data (for other species)
pm25_data = aq.extract_raw_data(start_date_data,end_date_data,param_code=88101)
CO_data = aq.extract_raw_data(start_date_data,end_date_data,param_code=42101)
other_data = pd.concat([pm25_data,CO_data])
other_data = other_data.set_index(pd.Series(data=range(len(other_data)))) # reindex to get rid of duplicate indices (index here is not significant)

# filter out stations that are definitley too far away to be of any use.
# we know nothing farther than r_max_interp+r_max_ML will be used.
# not necessary, but it should save time.
all_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,all_data,start_date_data,end_date_data,ignore_closest=False)
other_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,other_data,start_date_data,end_date_data,ignore_closest=False)
all_data = all_data.sort_values('Date Local')
other_data = other_data.sort_values('Date Local')

'''
Run the air quality estimation algorithms.
'''

# First run it after filling in missing data from nearby stations.
composite_data, orig, stations, station_obj_list = aq.create_composite_dataset(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data.copy(),other_data.copy(),ignore_closest=False)
data = aq.predict_aq_vals(composite_data,stations)

# Then run it using only the reported data.
nearby_stations = aq.identify_nearby_stations(latlon,r_max_interp,all_data.copy(),start_date,end_date,ignore_closest=False)
nearby_stations = aq.addon_stationid(nearby_stations)
nearby_stations = aq.remove_dup_stations(nearby_stations,ignore_closest=False)    
nearby_data = aq.extract_nearby_values(nearby_stations,all_data.copy(),start_date,end_date)
data_no_ML = aq.spatial_interp_variable_weights(nearby_data,nearby_stations,max_stations=10)

'''
Plot the results.
'''

# Plot comparing the results with and without
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(data_no_ML,color='gray',lw=1,label='Without data imputed')
ax.plot(data,color='b',lw=2,label='With data imputed')
ax.legend()
plt.show()