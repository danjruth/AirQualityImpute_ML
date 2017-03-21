# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:25:42 2017

@author: danjr
"""

import AQ_ML as aq

alldata = aq.addon_stationid(aq.all_data)

df=aq.identify_nearby_stations([30,-100],200)

for station_id in df.index:
    print(aq.identify_sampling_rate(station_id,aq.all_data))
    
    
station_df = pd.read_csv('aqs_monitors.csv')
