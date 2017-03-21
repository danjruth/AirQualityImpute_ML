# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:14:39 2017

@author: danjr
"""

import numpy as np
import pandas as pd
#import pickle


#station_df = pd.read_csv('aqs_monitors.csv')
#all_data = pd.read_csv('daily_81102_allYears.csv',usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean'])
#all_data = all_data.rename(columns={'Site Num':'Site Number'})


R_earth =  6371.0 # [km]
param_code = 81102


    
#station_df_2 = addon_stationid(station_df)
    



# distance in kilometers between two coordinates
def lat_lon_dist(point1,point2):
        
    # http://andrew.hedges.name/experiments/haversine/
    
    lat1 = point1[0]
    lon1 = point1[1]
    lat2 = point2[0]
    lon2 = point2[1]

    dlon = lon2-lon1
    dlat = lat2-lat1
    
    a=np.sin(np.deg2rad(dlat/2))**2 + np.cos(np.deg2rad(lat1))*np.cos(np.deg2rad(lat2))*np.sin(np.deg2rad(dlon/2))**2
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))        
    d = c*R_earth
    
    return d
    

def identify_sampling_rate(station_id,all_data,start_date=None,end_date=None):
    
    all_data = addon_stationid(all_data)
    station_data = all_data[all_data['station_ids']]
    station_data = station_data['station_ids'==station_id]
    station_data = station_data.set_index('Date Local')
    
    return station_data

# with a given latlon and r_max, pick out stations within that radius from a df
# with STATION DATA, not the metadata spreadsheet. This way we actually get 
# sites that have data
def identify_nearby_stations(latlon,r_max,df):
    
    # separate latitude/longitude
    my_lat = latlon[0]
    my_lon = latlon[1]
    
    # only look at stations reporting the parameter we're interested in
    param_stations = df.ix[df['Parameter Code']==param_code,:]    

    # compute distance between these sites and our point
    d = lat_lon_dist([param_stations['Latitude'],param_stations['Longitude']],[my_lat,my_lon])
    print(d)
    param_stations['Distance'] = d
    param_stations = param_stations.sort_values(['Distance'],ascending=True)
    
    # get rid of stations that are far away
    param_stations = param_stations[param_stations['Distance']<=r_max]

    return param_stations
    
    
# create a column of station ids
def addon_stationid(df):
    # create column of station ids. this will be the index
    station_ids = pd.Series(index=df.index)
    for i in station_ids.index:
        station_ids.ix[i] = str(df.ix[i]['State Code'])+'_'+str(df.ix[i]['County Code'])+'_'+str(df.ix[i]['Site Number'])
        print(i,len(station_ids.index))
    df['station_ids'] = station_ids    
    
    return df
    
# remove duplicate stations based on the station id (already created)
def remove_dup_stations(param_stations):
    
    # make the IDS the index, and get rid of duplicates
    param_stations = param_stations.set_index('station_ids')
    param_stations = param_stations[~param_stations.index.duplicated(keep='first')]

    print(param_stations)
    
    return param_stations
    
# pick out the values from stations nearby
def extract_nearby_values(stations,all_data):
    
    for idx in stations.index:
        
        county_code = stations.loc[idx]['County Code']
        state_code = stations.loc[idx]['State Code']
        site_number = stations.loc[idx]['Site Number']
        
        site_rawdata = all_data[(all_data['County Code']==county_code)&(all_data['State Code']==state_code)&(all_data['Site Number']==site_number)]
        site_rawdata = site_rawdata.set_index(site_rawdata['Date Local'])
        site_series = pd.Series(index=site_rawdata.index,data=site_rawdata['Arithmetic Mean'])
        site_series = site_series.rename(idx)