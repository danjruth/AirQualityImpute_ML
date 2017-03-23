# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:14:39 2017

@author: danjr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import pickle

#station_df_path = 'C:\Users\druth\Documents\FS\AirQuality\\aqs_monitors.csv'
station_df_path = 'C:\Users\danjr\Documents\ML\Air Quality\\aqs_monitors.csv'
#all_data_path = 'C:\Users\druth\Documents\FS\AirQuality\\daily_81102_allYears.csv'
all_data_path = 'C:\Users\danjr\Documents\ML\Air Quality\\daily_81102_allYears.csv'

#station_df = pd.read_csv()
#all_data = pd.read_csv(,usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean'])
#all_data = all_data.rename(columns={'Site Num':'Site Number'})


R_earth =  6371.0 # [km]
param_code = 81102


    
#station_df_2 = addon_stationid(station_df)

# class for a station that'll have data imputed
class aq_station:
    def __init__(self,station_id):
        self.station_data_series = pd.Series()
        self.nearby_stations = pd.DataFrame()
        self.nearby_data_df = pd.DataFrame() # each column is measurements from a different station
        self.station_info = pd.DataFrame()
        self.latlon = None
        self.station_id = station_id
        self.start_date = None
        self.end_date = None
        
    def get_station_data(self,r_max,df):
        self.nearby_stations = identify_nearby_stations(self.latlon,r_max,df)
        self.nearby_data_df = extract_nearby_values(self.nearby_stations,df,self.start_date,self.end_date)



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
    early = pd.to_datetime(good_dates[1:])
    later = pd.to_datetime(good_dates[0:-1])
    #print(early)
    #print(later)
    diff_data = early-later
    #print(diff_data)
    diff_period = pd.Series(index=good_dates[0:-1],data=diff_data)
    #diff_period = pd.Series(index=good_dates[0:-1],data=)
    
    #print(diff_period)
    
    '''
    plt.figure()
    plt.plot(diff_period/pd.Timedelta('1d'),'o')
    plt.show()
    '''
    
    estimated_rate = pd.Timedelta(np.median(diff_period))
    
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
def identify_nearby_stations(latlon,r_max,df,ignore_closest=False):
    
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
    
    if ignore_closest:
        param_stations = param_stations.loc[1:,:]

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

    #print(param_stations)
    
    return param_stations
    
# pick out the values from stations nearby
def extract_nearby_values(stations,all_data,start_date,end_date):
    
    print('extracting!')
        
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
        
        if ~site_series.isnull().all():
            df = pd.concat([df,site_series],axis=1)
        
    return df
    
    
def split_fill_unfill_stations(df):
    
    good_stations = pd.DataFrame()
    bad_stations = pd.DataFrame()
    for column in df:
        col_vals = df[column]
        #print(col_vals)
        rate = identify_sampling_rate(col_vals)
        print(rate)
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
    
def split_known_unknown_rows(predictors,site):
    
    # split up known rows from unkown rows
    have_out_vals = np.isnan(site)
    have_out_vals = np.where(have_out_vals==False)[0]
    need_out_vals = ~np.isnan(site)
    need_out_vals = np.where(need_out_vals==False)[0]
    
    known_x = predictors[have_out_vals,:]
    known_y = site[have_out_vals]
    unknown_x = predictors[need_out_vals,:]
    
    return known_x,known_y,unknown_x
    
  
def create_model_for_site(predictors,site):
    
    known_x,known_y,unknown_x = split_known_unknown_rows(predictors,site)
    if (len(known_y)<5 or len(unknown_x)<5):
        return None
    
    # shuffle rows
    from sklearn.utils import shuffle
    known_x_noshuffle = known_x
    known_y_noshuffle = known_y
    known_x,known_y = shuffle(known_x,known_y)
    known_y = known_y.ravel()
    
    # split known into test/train
    num_known = len(known_y)
    train_indx = range(0,int(num_known*.75))
    test_indx = range(int(num_known*.75),num_known)
    

    '''
    # linear model
    import sklearn.linear_model
    model = sklearn.linear_model.LinearRegression()
    model.fit(known_x[train_indx,:], known_y[train_indx])
    '''


    # neural network
    import sklearn.neural_network
    hl_size = (3,2)
    model = sklearn.neural_network.MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(hl_size),activation='relu')
    model.fit(known_x[train_indx,:], known_y[train_indx])
 

    
    # test the model
    model_predicted = model.predict(known_x[test_indx])
    model_known_predicted = model.predict(known_x[train_indx])
    
    # r2 score
    from sklearn.metrics import r2_score
    r2_predicted = r2_score(known_y[test_indx],model_predicted)
    r2_known_predicted = r2_score(known_y[train_indx],model_known_predicted)
    
    # target vs predicted
    plt.figure()
    plt.plot(known_y[test_indx],model_predicted,'.',label='Linear model',color='b')
    plt.plot(known_y[train_indx],model_known_predicted,'x',color='b')
    plt.plot([0, np.max(known_y)],[0, np.max(known_y)],color='k')
    plt.xlabel('Target')
    plt.ylabel('Predicted')
    plt.legend(loc=4)
    plt.title(str(r2_predicted)+', '+str(r2_known_predicted))
    plt.show()
    plt.pause(1)
    plt.show()
    
    #model = linear_model
    
    return model


def fill_with_model(predictors,site,model):
    
    if model is None:
        return site
    
    known_x,known_y,unknown_x = split_known_unknown_rows(predictors,site)
    
    predicted_y = model.predict(unknown_x)
    
    print(predicted_y)
    
    site[pd.isnull(site)] = predicted_y
    return site



def plot_station_locs(aq_obj):
    
    if aq_obj is None:
        print('No data here')
        return
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    
    ###############################
    # http://www.geophysique.be/2011/02/20/matplotlib-basemap-tutorial-09-drawing-circles/ 
    
    def shoot(lon, lat, azimuth, maxdist=None):
        """Shooter Function
        Original javascript on http://williams.best.vwh.net/gccalc.htm
        Translated to python by Thomas Lecocq
        """
        glat1 = lat * np.pi / 180.
        glon1 = lon * np.pi / 180.
        s = maxdist / 1.852
        faz = azimuth * np.pi / 180.
     
        EPS= 0.00000000005
        if ((np.abs(np.cos(glat1))<EPS) and not (np.abs(np.sin(faz))<EPS)):
            alert("Only N-S courses are meaningful, starting at a pole!")
     
        a=6378.13/1.852
        f=1/298.257223563
        r = 1 - f
        tu = r * np.tan(glat1)
        sf = np.sin(faz)
        cf = np.cos(faz)
        if (cf==0):
            b=0.
        else:
            b=2. * np.arctan2 (tu, cf)
     
        cu = 1. / np.sqrt(1 + tu * tu)
        su = tu * cu
        sa = cu * sf
        c2a = 1 - sa * sa
        x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
        x = (x - 2.) / x
        c = 1. - x
        c = (x * x / 4. + 1.) / c
        d = (0.375 * x * x - 1.) * x
        tu = s / (r * a * c)
        y = tu
        c = y + 1
        while (np.abs (y - c) > EPS):
     
            sy = np.sin(y)
            cy = np.cos(y)
            cz = np.cos(b + y)
            e = 2. * cz * cz - 1.
            c = y
            x = e * cy
            y = e + e - 1.
            y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *
                  d / 4. - cz) * sy * d + tu
     
        b = cu * cy * cf - su * sy
        c = r * np.sqrt(sa * sa + b * b)
        d = su * cy + cu * sy * cf
        glat2 = (np.arctan2(d, c) + np.pi) % (2*np.pi) - np.pi
        c = cu * cy - su * sy * cf
        x = np.arctan2(sy * sf, c)
        c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
        d = ((e * cy * c + cz) * sy * c + y) * sa
        glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2*np.pi)) - np.pi    
     
        baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)
     
        glon2 *= 180./np.pi
        glat2 *= 180./np.pi
        baz *= 180./np.pi
     
        return (glon2, glat2, baz)
    
    def equi(m, centerlon, centerlat, radius, *args, **kwargs):
        glon1 = centerlon
        glat1 = centerlat
        X = []
        Y = []
        for azimuth in range(0, 360):
            glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
            X.append(glon2)
            Y.append(glat2)
        X.append(X[0])
        Y.append(Y[0])
     
        #m.plot(X,Y,**kwargs) #Should work, but doesn't...
        X,Y = m(X,Y)
        plt.plot(X,Y,**kwargs)
        
    #######################################
    
    # "unpack" data from air quality object
    #print(aq_obj.monitor_info)
    monitor_info = aq_obj.monitor_info
    my_lon = aq_obj.lat_lon[1]
    my_lat = aq_obj.lat_lon[0]
    r_max = aq_obj.r_max
    param_code = aq_obj.param
    #site_name = aq_obj.site_name
    
    num_stations = len(aq_obj.station_list)
    
    # create colors to correspond to each of the stations
    RGB_tuples = [(x/num_stations, (1-x/num_stations),.5) for x in range(0,num_stations)]
    color_dict = {}
    for x in range(0,num_stations):
        color_dict[aq_obj.station_list.index[x]] = RGB_tuples[x]
    
    # set viewing window for map plot
    scale_factor = 60.0 # lat/lon coords to show from center point per km of r_max
    left_lim = my_lon-r_max/scale_factor
    right_lim = my_lon+r_max/scale_factor
    bottom_lim = my_lat-r_max/scale_factor
    top_lim = my_lat+r_max/scale_factor
    
    
    fig = plt.figure(figsize=(20, 12), facecolor='w')    
    
    plt.subplot(1,2,1)
    m = Basemap(projection='merc',resolution='c',lat_0=my_lat,lon_0=my_lon,llcrnrlon=left_lim,llcrnrlat=bottom_lim,urcrnrlon=right_lim,urcrnrlat=top_lim)
    m.shadedrelief()
    m.drawstates()
    m.drawcountries()
    m.drawrivers()
    m.drawcoastlines()
    
    # plot each EPA site on the map, and connect it to the soiling station with a line whose width is proportional to the weight
    for i in range(0,len(aq_obj.station_list)):
        
        plt.subplot(1,2,1)
        (x,y) = m([aq_obj.station_list.iloc[i]['Longitude'],my_lon],[aq_obj.station_list.iloc[i]['Latitude'],my_lat])
        m.plot(x,y,color = RGB_tuples[i])
        
        (x,y) = m(aq_obj.station_list.iloc[i]['Longitude'],aq_obj.station_list.iloc[i]['Latitude'])
        #m.plot(x,y,'o',color = RGB_tuples[i])        
        plt.text(x,y,str(i))
        
    # finish the map plot by putting the soiling station loc and radius on there
    plt.subplot(1,2,1)
    (x,y) = m(my_lon,my_lat)
    m.plot(x,y,'kx',markersize=20,lw=3)
    equi(m, my_lon, my_lat, r_max,color='k')        
    plt.title('Found '+str(num_stations)+' acceptable EPA '+str(param_code)+' stations within '+str(r_max)+' km')        
    plt.show()
    
    '''
    # readings for all the stations
#    ax=plt.subplot(2,3,2)
#    print(aq_obj.station_readings)
#    for station in aq_obj.station_readings.columns.values:
#        ax.plot(aq_obj.station_readings[station],'x-',label=station)
#    ax.set_ylabel('Station Reading')
#    plt.show()    
    print(color_dict)
    # weights for all the stations
    ax=plt.subplot(2,2,2)
    #print(aq_obj.station_weights)
    for station in aq_obj.station_weights.columns.values:
        relative_weights = aq_obj.station_weights[station]/aq_obj.station_weights.sum(axis=1)*100
        #print('Here are the relative weights:')
        #print(relative_weights)
        ax.plot(relative_weights,'x-',label=station,color = color_dict[station])
    ax.set_ylabel('Station Weight [%]')
    ax.set_xlabel('Distance [km]')
    plt.show()  
    
    # plot each station reading profile
    for station in aq_obj.station_readings.columns.values:
        plt.subplot(2,2,4)
        plt.plot(aq_obj.station_readings[station],'x-',label=station,color = color_dict[station])
    plt.show()    
        
    # plot averaged value
    plt.subplot(2,2,4)
    plt.plot(aq_obj.daily_average,'.-',label='Weighted Average',lw=2,color='k')
    plt.ylabel('Concentration')
    #plt.legend()
    plt.show()
    '''

    
    return fig