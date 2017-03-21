# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:14:39 2017

@author: danjr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pickle

station_df_path = 'C:\Users\druth\Documents\FS\AirQuality\\aqs_monitors.csv'
all_data_path = 'C:\Users\druth\Documents\FS\AirQuality\\daily_81102_allYears.csv'


#station_df = pd.read_csv()
#all_data = pd.read_csv(,usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean'])
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
    
def identify_sampling_rate(series):
    is_nan = pd.isnull(series)
        
    good_dates = series.index[is_nan==False]
    diff_period = pd.Series(index=good_dates[0:-1],data=(good_dates[1:]-good_dates[0:-1]))
    
    print(diff_period)
    
    '''
    plt.figure()
    plt.plot(diff_period/pd.Timedelta('1d'),'o')
    plt.show()
    '''
    
    estimated_rate = diff_period.median()
    
    return estimated_rate
    
    
'''
def identify_sampling_rate(station_id,all_data,start_date=None,end_date=None):
    
    all_data = addon_stationid(all_data)
    station_data = all_data[all_data['station_ids']]
    station_data = station_data['station_ids'==station_id]
    station_data = station_data.set_index('Date Local')
    
    return station_data
'''

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
def extract_nearby_values(stations,all_data,start_date,end_date):
    
    #times = pd.date_range(start=start_date,end=end_date)
    
    df = pd.DataFrame()
    
    # collect data for each nearby station
    for idx in stations.index:
        
        county_code = stations.loc[idx]['County Code']
        state_code = stations.loc[idx]['State Code']
        site_number = stations.loc[idx]['Site Number']
        
        site_rawdata = all_data[(all_data['County Code']==county_code)&(all_data['State Code']==state_code)&(all_data['Site Number']==site_number)]
        site_rawdata = site_rawdata.set_index(pd.to_datetime(site_rawdata['Date Local']))
        
        site_rawdata = site_rawdata[(site_rawdata.index>=start_date)&(site_rawdata.index<=end_date)]
        
        site_rawdata = site_rawdata[~site_rawdata.index.duplicated(keep='first')]
        site_series = pd.Series(index=site_rawdata.index,data=site_rawdata['Arithmetic Mean'])
        site_series = site_series.rename(idx)
        
        df = pd.concat([df,site_series],axis=1)
        
    return df
    
    
def split_fill_unfill_stations(df):
    
    good_stations = pd.DataFrame()
    bad_stations = pd.DataFrame()
    for column in df:
        col_vals = df[column]
        rate = identify_sampling_rate(col_vals)
        num_missing = len(col_vals[pd.isnull(col_vals)==True])
        portion_missing = float(num_missing)/float(len(col_vals))
        print(num_missing,len(col_vals),portion_missing)
  
        enough_data = (rate==pd.Timedelta('1d')) & (portion_missing < 0.1)
        if enough_data:
            good_stations = pd.concat([good_stations,col_vals],axis=1)
        else:
            bad_stations = pd.concat([bad_stations,col_vals],axis=1)
            
    return good_stations, bad_stations
    
def fill_missing_predictors(predictors):    
    
    import sklearn.preprocessing
    imp = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    predictors = imp.fit_transform(predictors)
    
    return predictors
  
def create_model_for_site(predictors,site):
    
    
    
    # split up known rows from unkown rows
    have_out_vals = np.isnan(site)
    have_out_vals = np.where(have_out_vals==False)[0]
    need_out_vals = ~np.isnan(site)
    need_out_vals = np.where(need_out_vals==False)[0]
    num_known = len(have_out_vals)    
    known_x = predictors[have_out_vals,:]
    known_x = known_x
    known_y = site[have_out_vals]
    
    print('known x:')
    print(known_x)
    print('known y:')
    print(known_y)
    
    # shuffle rows
    from sklearn.utils import shuffle
    known_x_noshuffle = known_x
    known_y_noshuffle = known_y
    known_x,known_y = shuffle(known_x,known_y)
    known_y = known_y.ravel()
    
    train_indx = range(0,int(num_known*.75))
    test_indx = range(int(num_known*.75),num_known)
    
    
    
    # create/fit model
    
    import sklearn.linear_model
    linear_model = sklearn.linear_model.LinearRegression()
    linear_model.fit(known_x[train_indx,:], known_y[train_indx])
    
    # test the model
    linear_predicted = linear_model.predict(known_x[test_indx])
    linear_known_predicted = linear_model.predict(known_x[train_indx])
    
    # target vs predicted
    plt.figure()
    plt.plot(known_y[test_indx],linear_predicted,'.',label='Linear model',color='b')
    plt.plot(known_y[train_indx],linear_known_predicted,'x',color='b')
    plt.plot([0, np.max(known_y)],[0, np.max(known_y)],color='k')
    plt.xlabel('Target')
    plt.ylabel('Predicted')
    plt.legend(loc=4)
    plt.title('Model performance')
    plt.show()
    
    model = linear_model
    
    return model


#def fill_with_model(predictors,site,model):