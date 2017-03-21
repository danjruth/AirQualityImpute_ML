# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:07:18 2017

@author: danjr
"""

import AQ_ML as aq
import pandas as pd
import numpy as np

#station_df = pd.read_csv('aqs_monitors.csv')

#all_data = pd.read_csv('daily_81102_allYears.csv',usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean','Parameter Code','Latitude','Longitude'])
#all_data = all_data.rename(columns={'Site Num':'Site Number'})

latlon = (30,-90)
r_max = 100

start_date = '2012-01-01'
end_date = '2013-01-01'

station_data = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='1D').date)

#stations = aq.identify_nearby_stations(latlon,r_max,all_data)


# for each station identified, create a column of a dataframe with values for that date
for idx in stations.index:
    
    county_code = stations.loc[idx]['County Code']
    state_code = stations.loc[idx]['State Code']
    site_number = stations.loc[idx]['Site Number']
    
    site_rawdata = all_data[(all_data['County Code']==county_code)&(all_data['State Code']==state_code)&(all_data['Site Number']==site_number)]
    site_rawdata = site_rawdata.set_index(site_rawdata['Date Local'])
    site_series = pd.Series(index=site_rawdata.index,data=site_rawdata['Arithmetic Mean'])
    site_series = site_series.rename(idx)
    
    station_data[
    
   # filtered_rawdata = all_data['Site Number'==]