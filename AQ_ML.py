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

# some constants
R_earth =  6371.0 # [km]
param_code = 81102

# plot a matrix of nearby station values, labeling each station
def matrix_val_plot(df,fig=None):
    if fig==None:
        fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(df.copy().transpose(),aspect='auto')
    ax.set_yticklabels(df.columns.values)
    ax.set_yticks(range(0,len(df.columns.values)))
    return fig

# class for a station that'll have data imputed
class aq_station:
    def __init__(self,station_id,ignoring=None):
        self.station_data_series = pd.Series()
        self.nearby_stations = pd.DataFrame()
        self.gs = pd.DataFrame()
        self.bs = pd.DataFrame()
        self.nearby_data_df = pd.DataFrame() # each column is measurements from a different station
        self.station_info = pd.DataFrame()
        self.latlon = None
        self.station_id = station_id
        self.start_date = None
        self.end_date = None
        self.ignoring=ignoring
        
    def get_station_data(self,r_max,df):
        self.nearby_stations = identify_nearby_stations(self.latlon,r_max,df)
        self.nearby_stations = addon_stationid(self.nearby_stations)
        self.nearby_stations = remove_dup_stations(self.nearby_stations,ignore_closest=False)
        if self.ignoring is not None:
            self.nearby_stations = self.nearby_stations[self.nearby_stations.index!=self.ignoring].copy()
        self.nearby_data_df = extract_nearby_values(self.nearby_stations,df,self.start_date,self.end_date)
        self.this_station = pd.Series(self.nearby_data_df.iloc[:,0]).copy()
#        fig = matrix_val_plot(self.nearby_data_df.copy())
#        fig.suptitle('Getting station data. Here is all nearby data AND the known data (first row).')
#        fig.show()
        self.nearby_data_df = self.nearby_data_df.iloc[:,1:].copy()
        
    def create_model(self):
        self.gs,bs = split_fill_unfill_stations(self.nearby_data_df)        
        self.gs = fill_missing_predictors(self.gs)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.matshow(self.gs.copy().transpose(),aspect='auto')
        ax.set_yticklabels(self.gs.columns.values)
        ax.set_yticks(range(0,len(self.gs.columns.values)))
        fig.suptitle('Creating a model. These are the "good stations".')
        fig.show()
        self.model = create_model_for_site(self.gs,self.this_station)
        
    def run_model(self):        
        self.composite_data = fill_with_model(self.gs,self.this_station.copy(),self.model)
#        plt.figure()
#        plt.plot(self.this_station.copy(),'x',markersize=5)
#        plt.plot(self.composite_data)
#        plt.title('Filled in')
#        plt.show()



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
    
# look at data from an EPA station and guess if it's on the 1, 3, or 6 day schedule
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
def remove_dup_stations(param_stations,ignore_closest=False):
    
    # make the IDS the index, and get rid of duplicates
    param_stations = param_stations.set_index('station_ids')
    param_stations = param_stations[~param_stations.index.duplicated(keep='first')]
    
    if ignore_closest:
        param_stations = param_stations.iloc[1:,:]

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
    
# for a given set of stations, separate the ones that are full enough and those that aren't
# only the full ones will be used
def split_fill_unfill_stations(df):
    
    good_stations = pd.DataFrame()
    bad_stations = pd.DataFrame()
    
    # look at each column (data for a given station) and see if it's good or bad
    for column in df:
        col_vals = df[column]
        rate = identify_sampling_rate(col_vals)
        num_missing = len(col_vals[pd.isnull(col_vals)==True])
        portion_missing = float(num_missing)/float(len(col_vals))
  
        # criteria for using the site: mostly daily and not missing much
        enough_data = (rate==pd.Timedelta('1d')) & (portion_missing < 0.2)
        if enough_data:
            good_stations = pd.concat([good_stations,col_vals],axis=1)
        else:
            bad_stations = pd.concat([bad_stations,col_vals],axis=1)
            
    return good_stations, bad_stations
    
# fill in missing predictor values, keeping it as a df
def fill_missing_predictors(predictors):    
    
    print(type(predictors))
    
    import sklearn.preprocessing
    imp = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    predictors_filled = imp.fit_transform(predictors).copy()
    predictors_filled_df = pd.DataFrame(index=predictors.index,columns=predictors.columns,data=predictors_filled)
    
    print(type(predictors_filled_df))
    
    return predictors_filled_df
    
# based on which values for the site are available, split up predictors/known
# into known and unknown
def split_known_unknown_rows(predictors,site):
    
    # split up known rows from unkown rows
    have_out_vals = np.isnan(site)
    have_out_vals = np.where(have_out_vals==False)[0]
    need_out_vals = ~np.isnan(site)
    need_out_vals = np.where(need_out_vals==False)[0]
    
    known_x = predictors.iloc[have_out_vals,:].copy()
    known_y = site[have_out_vals].copy()
    unknown_x = predictors.iloc[need_out_vals,:].copy()
    
    return known_x,known_y,unknown_x
    
# given training data, create a model that'll be used to predict the missing data
def create_model_for_site(predictors,site):
    
    known_x,known_y,unknown_x = split_known_unknown_rows(predictors,site)
    if (len(known_y)<5 or len(unknown_x)<5):
        return None
    
    # shuffle rows
    from sklearn.utils import shuffle
    known_x,known_y = shuffle(known_x.copy(),known_y.copy())
    known_y = known_y.ravel()
    
    # split known into test/train
    num_known = len(known_y)
    train_indx = range(0,int(num_known*.75))
    test_indx = range(int(num_known*.75),num_known)
    

    # linear model
    import sklearn.linear_model
    model = sklearn.linear_model.LinearRegression()

    '''
    # neural network
    import sklearn.neural_network
    hl_size = (2)
    model = sklearn.neural_network.MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(hl_size),activation='relu')
    '''

    model.fit(known_x.iloc[train_indx,:], known_y[train_indx])
    
    # test the model
    model_predicted = model.predict(known_x.iloc[test_indx])
    model_known_predicted = model.predict(known_x.iloc[train_indx])
    
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
    
    print(str(r2_predicted)+', '+str(r2_known_predicted))
        
    return model

# use the model to fill the missing data, returning a "composite" series
def fill_with_model(predictors,site,model):
    
    if model is None:
        return site
    
    # split known/unknown, simulate
    known_x,known_y,unknown_x = split_known_unknown_rows(predictors,site)    
    predicted_y = model.predict(unknown_x)
        
    # replace missing with the simulated, returning the composite
    composite_series = site.copy() # start with site data
    composite_series[pd.isnull(site)] = predicted_y.copy()
    return composite_series
    
    
# once nearby stations have been picked, add on a column of their weights (for
# the spatial interpolation algorithm)
def create_station_weights(nearby_metadata):
    
    # determine the weighting for the stations
    station_weights = pd.Series(index=nearby_metadata.index)
    num_stations = len(nearby_metadata)
    for station in nearby_metadata.index:
        # average distance between this site and others
        total_dist = 0
        for other_station in nearby_metadata.index:
            if station != other_station:
                dist_between_stations = lat_lon_dist([nearby_metadata.loc[station]['Latitude'],nearby_metadata.loc[station]['Longitude']],[nearby_metadata.loc[other_station]['Latitude'],nearby_metadata.loc[other_station]['Longitude']])
                total_dist = total_dist + dist_between_stations        
            # average distance between this and other stations
            r_jk_bar = total_dist/(num_stations-1)
        
        CW_ijk = 1/float(num_stations) + r_jk_bar/nearby_metadata.loc[station]['Distance']    
        R_ij = (1/nearby_metadata.loc[station]['Distance'] )**2    
        station_weights[station] = R_ij * CW_ijk
        
    nearby_metadata['weight'] = station_weights
    
    return nearby_metadata

# take a df of nearby data, and metadata df that has station weights, to interp
def spatial_interp(nearby_data,nearby_metadata):
        
    dates = nearby_data.index
    data = pd.Series(index=dates)
    
    # perform weighted average of stations for this day 
    for date in dates:
        weights_sum = 0
        values_sum = 0
        for station in nearby_metadata.index:
            if pd.notnull(nearby_data.loc[date,station]):
                weights_sum = weights_sum + nearby_metadata.loc[station,'weight']
                values_sum = values_sum + nearby_data.loc[date,station]*nearby_metadata.loc[station,'weight']

        if weights_sum is not 0: # avoid dividing by zero--if no data for any of them, keep it as NaN
            data[date] = values_sum/weights_sum
        else:
            data[date] = np.nan
            
    return data
    
# plot the original and composite data for each nearby station as well as the 
# interpolated value
def final_big_plot(data,orig,composite,nearby_metadata):
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111)
    
    weights = nearby_metadata['weight']
    w_lims = (weights.min(),weights.max())
    w_range = w_lims[1] - w_lims[0]
    
    for station in nearby_metadata.index:
        w = nearby_metadata.loc[station,'weight']
        p = (w-w_lims[0])/w_range
        ax.plot(orig.loc[:,station],'o',color=(p,0,0))
        ax.plot(composite.loc[:,station],'--',lw=1,color=(p,0,0))
        
    ax.plot(data,color='b',lw=2)
    
    

    

# plot each station on a basemap
def plot_station_locs(stations):
    
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
    #monitor_info = aq_obj.monitor_info
    my_lon = stations['Longitude'][0]
    my_lat = stations['Latitude'][0]
    r_max = 150
    #param_code = aq_obj.param
    #site_name = aq_obj.site_name
    
    num_stations = len(stations)
    
    # create colors to correspond to each of the stations
    RGB_tuples = [(x/num_stations, (1-x/num_stations),.5) for x in range(0,num_stations)]
    color_dict = {}
    for x in range(0,num_stations):
        color_dict[stations.index[x]] = RGB_tuples[x]
    
    # set viewing window for map plot
    scale_factor = 60.0 # lat/lon coords to show from center point per km of r_max
    left_lim = my_lon-r_max/scale_factor
    right_lim = my_lon+r_max/scale_factor
    bottom_lim = my_lat-r_max/scale_factor
    top_lim = my_lat+r_max/scale_factor
    
    
    fig = plt.figure(figsize=(20, 12), facecolor='w')    
    
    m = Basemap(projection='merc',resolution='c',lat_0=my_lat,lon_0=my_lon,llcrnrlon=left_lim,llcrnrlat=bottom_lim,urcrnrlon=right_lim,urcrnrlat=top_lim)
    m.shadedrelief()
    m.drawstates()
    m.drawcountries()
    m.drawrivers()
    m.drawcoastlines()
    
    plt.show()
    
    # plot each EPA site on the map, and connect it to the soiling station with a line whose width is proportional to the weight
    for i in range(0,num_stations):
        
        (x,y) = m(stations.iloc[i]['Longitude'],stations.iloc[i]['Latitude'])
        m.plot(x,y,'o',color = RGB_tuples[i])
        
        (x,y) = m(stations.iloc[i]['Longitude'],stations.iloc[i]['Latitude'])
        plt.text(x,y,stations.index[i])

    
    return fig    
    
# do everything to get air quality data
def predict_aq_vals(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data,ignore_closest=False):
    
    # this will store the metadata for each station that'll be used
    stations = identify_nearby_stations(latlon,r_max_interp,all_data.copy())
    stations = addon_stationid(stations)
    stations = remove_dup_stations(stations)
    print(stations)
    
    # get rid of the closest station if you want to use that for validation
    # also save its reading so you can compare later
    closest = None # name of the closest station to ignore
    if ignore_closest:
        closest = stations.index[0]
        print(closest)
        
        closest_obj = aq_station(closest)
        closest_obj.latlon = (stations.loc[closest,'Latitude'],stations.loc[closest,'Longitude'])
        closest_obj.start_date = start_date
        closest_obj.end_date = end_date
        closest_obj.get_station_data(r_max_ML,all_data.copy())
        
        target_data = closest_obj.this_station
        
        stations = stations.iloc[1:,:]
        
    print(stations)
    
    stations = create_station_weights(stations)
    
    # plot these stations on a map
    plot_station_locs(stations)
    
    orig = pd.DataFrame(columns=stations.index.copy())
    
    # for each nearby station, fill in missing data
    composite_data = pd.DataFrame()
    for station in stations.index:
    
        station_obj = None
        
        station_obj =aq_station(station,ignoring=closest)
        station_obj.latlon = (stations.loc[station,'Latitude'],stations.loc[station,'Longitude'])
        station_obj.start_date = start_date
        station_obj.end_date = end_date
        station_obj.get_station_data(r_max_ML,all_data.copy())
        orig.loc[:,station] = station_obj.this_station.copy()
        station_obj.create_model()
        station_obj.run_model()
        
        composite_data.loc[:,station] = station_obj.composite_data.rename(station).copy()
        
    data = spatial_interp(composite_data,stations)   
    final_big_plot(data,orig,composite_data,stations)
    
    matrix_val_plot(orig)
    matrix_val_plot(composite_data)
    
    if ignore_closest:
        
        # both time series
        plt.figure()
        plt.plot(data,label='predicted')
        plt.plot(target_data,'.-',label='target')
        plt.legend()
        plt.show()
        
        # one against the other
        plt.figure()
        plt.scatter(target_data,data)
        plt.plot([0,0],[data.max(),data.max()])
        plt.ylabel('Predicted')
        plt.xlabel('Target')
        plt.show()
        
        return data, target_data
    else:
        return data