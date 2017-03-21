# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:07:18 2017

@author: danjr
"""

import AQ_ML as aq
import pandas as pd
import numpy as np

station_df = pd.read_csv('aqs_monitors.csv')

all_data = pd.read_csv('daily_81102_allYears.csv',usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean','Parameter Code','Latitude','Longitude'])
all_data = all_data.rename(columns={'Site Num':'Site Number'})

latlon = (30,-90)
r_max = 100

start_date = '2012-01-01'
end_date = '2013-01-01'

station_data = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='1D').date)

stations = aq.identify_nearby_stations(latlon,r_max,all_data)

nearby_data = aq.extract_nearby_stations(stations,all_data)