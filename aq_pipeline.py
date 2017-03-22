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
if ~('all_data' in locals()):
    all_data = pd.read_csv(aq.all_data_path,usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean','Parameter Code','Latitude','Longitude'])
    all_data = all_data.rename(columns={'Site Num':'Site Number'})




latlon = (33.424564, -111.928001) # ASU
r_max = 100

start_date = '2013-03-01'
end_date = '2014-01-01'

station_data = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='1D').date)

stations = aq.identify_nearby_stations(latlon,r_max,all_data)
stations = aq.addon_stationid(stations)
stations = aq.remove_dup_stations(stations)




nearby_data = aq.extract_nearby_values(stations,all_data,start_date,end_date)

is_missing = pd.isnull(nearby_data)

# split up the stations in to good stations (enough data) and bad ones (to be imputed)
gs,bs = aq.split_fill_unfill_stations(nearby_data)

# initialize df that'll have the composite data
filled_all = nearby_data
'''
# replace missing data in predictors (won't be too many of these)
gs = aq.fill_missing_predictors(gs)

# show the good and bad stations
fig = plt.figure()
ax_g = fig.add_subplot(2,1,1)
ax_g.matshow(gs.transpose())
ax_b = fig.add_subplot(2,1,2)
ax_b.matshow(bs.transpose())

filled_bad = pd.DataFrame()

# fill in missing data in each "bad column"
for column in bs:
    col = bs[column]
    col_vals = pd.Series(index=col.index,data=col.values)
    model = aq.create_model_for_site(gs,col_vals)
    filled = aq.fill_with_model(gs,col_vals,model)
    filled_bad = pd.concat([filled_bad,filled],axis=1)
    
    filled_all[col.name] = filled
    
# show the filled/unfilled data
fig2 = plt.figure()
ax_u = fig2.add_subplot(2,1,1)
ax_u.matshow(bs.transpose())
ax_f = fig2.add_subplot(2,1,2)
ax_f.matshow(filled_bad.transpose())

plt.matshow(filled_all.transpose())
missing_t = is_missing.transpose()
ci=0
for column in missing_t:
    ri = 0
    print(column)
    for row in column:
        if row == True:
            plt.plot(ci,ri,'o',color=(1,1,1))
        ri = ri+1
    ci=ci+1    

plt.matshow(is_missing.transpose())
'''