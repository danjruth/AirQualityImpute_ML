# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:14:39 2017

@author: danjr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# some constants
R_earth =  6371.0 # [km]

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
        
    def get_station_data(self,r_max,df,other_data):
        print('----------------------')
        print('Getting station data for station '+self.station_id)
        
        # get data of interest
        self.nearby_stations = identify_nearby_stations(self.latlon,r_max,df,self.start_date,self.end_date)
        self.nearby_stations = addon_stationid(self.nearby_stations)
        self.nearby_stations = remove_dup_stations(self.nearby_stations,ignore_closest=False)
        if self.ignoring is not None:
            print('   Removing stations with latitude '+str(self.ignoring[0]))
            self.nearby_stations = self.nearby_stations[self.nearby_stations['Latitude']!=self.ignoring[0]].copy()
        self.nearby_data_df = extract_nearby_values(self.nearby_stations,df,self.start_date,self.end_date)
        if self.station_id in self.nearby_data_df.columns:
            self.this_station = pd.Series(self.nearby_data_df[self.station_id]).copy()
        else:
            self.this_station = pd.Series()
            print('No data for this station!')
        self.nearby_data_df = self.nearby_data_df.drop(self.station_id, axis=1) # get rid of the closest data: this is the target data, not used in training
        
        # get the data for the other stations
        self.other_stations = identify_nearby_stations(self.latlon,r_max,other_data,self.start_date,self.end_date)
        self.other_stations = addon_stationid(self.other_stations)
        self.other_stations = remove_dup_stations(self.other_stations,ignore_closest=False)
        if self.ignoring is not None:
            print('   Removing stations with latitude '+str(self.ignoring[0]))
            self.other_stations = self.other_stations[self.other_stations['Latitude']!=self.ignoring[0]].copy()
        self.other_data_df = extract_nearby_values(self.other_stations,other_data,self.start_date,self.end_date)
        
    def plot_matrix_station(self):
        
        fig = plt.figure(figsize=(12,6))
        
        first_day = self.nearby_data_df.index[0]

        ax1 = fig.add_subplot(211)
        days_array = np.arange((self.this_station.index[0]-first_day)/pd.Timedelta('1D'),(self.this_station.index[-1]-first_day)/pd.Timedelta('1D')+1)
        
        ax1.plot(self.this_station,'.-')
        ax1.set_ylabel(self.this_station.name)
        
        ax2 = fig.add_subplot(212) #,sharex=ax1
        ax2.matshow(self.gs.copy().transpose(),aspect='auto',extent=[0,len(days_array),0,len(self.gs.columns)],origin='lower')
        ax2.set_yticklabels(self.gs.columns.values)
        ax2.set_yticks(range(0,len(self.gs.columns.values)))
        
        return fig        
        
    def create_model(self):
        
        # determine which features should be used for this model
        self.gs,bs = feature_selection(pd.concat([self.nearby_data_df,self.other_data_df],axis=1),self.this_station) # nearby_data_df does NOT include the station to predict
            
        if self.gs.empty:
            print('No good sites found to make this model. No model being created...')
            self.model = None
            return
            
        # fill missing predictors
        self.gs = fill_missing_predictors(self.gs)
        
        # plot the features and the value to predict
        self.plot_matrix_station()
        
        # create a model
        self.model = create_model_for_site(self.gs,self.this_station)
        
    def run_model(self):        
        self.composite_data = fill_with_model(self.gs,self.this_station.copy(),self.model)

def extract_raw_data(start_date,end_date,param_code=81102):
    
    #folder = 'C:\Users\danjr\Documents\ML\Air Quality\data\\'
    folder = 'C:\Users\druth\Documents\epa_data\\'
    
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    years = np.arange(start_year,end_year+1)
    
    data = pd.DataFrame()
    for year in years:
        print(year)
        year_df = pd.read_csv(folder+'daily_'+str(param_code)+'_'+str(year)+'.csv',usecols=['State Code','County Code','Site Num','POC','Date Local','Arithmetic Mean','Parameter Code','Latitude','Longitude'])
        year_df = year_df.rename(columns={'Site Num':'Site Number'})
        data=pd.concat([data,year_df],ignore_index=True)
        
    return data

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
    
# look at data from an EPA station and guess if it's on the 1, 3, 6, or 12 day schedule
def identify_sampling_rate(series):
    
    is_nan = pd.isnull(series)        
    good_dates = series.index[is_nan==False]
    early = pd.to_datetime(good_dates[1:])
    later = pd.to_datetime(good_dates[0:-1])
    
    diff_data = early-later
    diff_period = pd.Series(index=good_dates[0:-1],data=diff_data)
    estimated_rate = pd.Timedelta(np.median(diff_period))
    
    return estimated_rate
    
# with a given latlon and r_max, pick out stations within that radius from a df
# with STATION DATA, not the metadata spreadsheet. This way we actually get 
# sites that have data
def identify_nearby_stations(latlon,r_max,df,start_date,end_date,ignore_closest=False):
    
    # separate latitude/longitude
    my_lat = latlon[0]
    my_lon = latlon[1]

    param_stations = df.copy()
    
    # compute distance between these sites and our point
    d = lat_lon_dist([param_stations['Latitude'],param_stations['Longitude']],[my_lat,my_lon])
    param_stations['Distance'] = d
    param_stations = param_stations.sort_values(['Distance'],ascending=True)
    
    # get rid of stations that are far away
    param_stations = param_stations[param_stations['Distance']<=r_max]
    
    # add the datetime, so sites without data in our date range can be excluded
    param_stations['Date Local'] = pd.to_datetime(param_stations['Date Local'])
    param_stations = param_stations[(param_stations['Date Local']>=start_date)&(param_stations['Date Local']<=end_date)]

    return param_stations    
    
# create a column of station ids
def addon_stationid(df):
    # create column of station ids. this will be the index
    station_ids = pd.Series(index=df.index)
    for i in station_ids.index:
        station_ids.loc[i] = str(df.loc[i]['Parameter Code'])+'_'+str(df.loc[i]['State Code'])+'_'+str(df.loc[i]['County Code'])+'_'+str(df.loc[i]['Site Number'])+'_'+str(df.loc[i]['POC'])
    df['station_ids'] = station_ids    
    
    return df
    
# remove duplicate stations based on the station id (already created)
def remove_dup_stations(param_stations,ignore_closest=False):
    
    # make the IDS the index, and get rid of duplicates
    param_stations = param_stations.set_index('station_ids')
    param_stations = param_stations[~param_stations.index.duplicated(keep='first')]
    
    if ignore_closest:
        param_stations = param_stations.iloc[1:,:]
    
    return param_stations
    
# pick out the values from stations nearby
def extract_nearby_values(stations,all_data,start_date,end_date):
    
    print('Extracting nearby values...')
        
    df = pd.DataFrame()
    
    # collect data for each nearby station
    for idx in stations.index:
        
        param_code = stations.loc[idx]['Parameter Code']
        county_code = stations.loc[idx]['County Code']
        state_code = stations.loc[idx]['State Code']
        site_number = stations.loc[idx]['Site Number']
        POC = stations.loc[idx]['POC']
        
        site_rawdata = all_data[(all_data['Parameter Code']==param_code)&(all_data['County Code']==county_code)&(all_data['State Code']==state_code)&(all_data['Site Number']==site_number)&(all_data['POC']==POC)]
        site_rawdata = site_rawdata.set_index(pd.to_datetime(site_rawdata['Date Local']))
        
        site_rawdata = site_rawdata[(site_rawdata.index>=start_date)&(site_rawdata.index<=end_date)]
        
        site_rawdata = site_rawdata[~site_rawdata.index.duplicated(keep='first')] # see if this is necessary
        site_series = pd.Series(index=site_rawdata.index,data=site_rawdata['Arithmetic Mean'])
        site_series = site_series.rename(idx)
        
        if ~site_series.isnull().all():
            df = pd.concat([df,site_series],axis=1)
        
    return df
    
# for a given set of stations, separate the ones that are full enough and those that aren't
# only the full ones will be used
def feature_selection(df,this_station,stations_to_keep=None):
    
    good_stations = pd.DataFrame()
    bad_stations = pd.DataFrame()
    
    missing_days = this_station.index[pd.isnull(this_station)]
                                      
    if len(missing_days)==0:
        missing_days = this_station.index

    # look at each column (data for a given station) and see if it's good or bad
    for column in df:
        col_vals = df[column]
        col_while_missing = col_vals[missing_days]
        rate = identify_sampling_rate(col_vals)
        num_missing = len(col_while_missing[pd.isnull(col_vals)==True])
        portion_missing = float(num_missing)/float(len(col_while_missing))
          
        # criteria for using the site: mostly daily and not missing much
        enough_data = (rate==pd.Timedelta('1d')) & (portion_missing < 0.1)
        if enough_data:
            good_stations = pd.concat([good_stations,col_vals],axis=1)
        else:
            bad_stations = pd.concat([bad_stations,col_vals],axis=1)
    
    print(str(len(good_stations.columns))+' good stations, '+str(len(bad_stations.columns))+' bad stations.')
    
    # choose how many stations to keep based on how many datapoints there will be to train on
    if stations_to_keep is None:
        stations_to_keep = min(10,max(3,int(len(this_station.index[pd.notnull(this_station)])/20)))
    
    corr_vals = pd.Series(index=good_stations.columns)
    for station in corr_vals.index:
        corr_vals[station] = good_stations[station].corr(this_station)
    corr_vals = corr_vals.sort_values(ascending=False)
    corr_vals = corr_vals[corr_vals>0]
    cols_to_keep = corr_vals.index.tolist()[0:min(stations_to_keep,len(corr_vals))]
    good_stations_filtered = good_stations.loc[:,cols_to_keep]
        
    return good_stations_filtered, bad_stations
    
# fill in missing predictor values, keeping it as a df
def fill_missing_predictors(predictors):    
    
    print('Filling in the missing values from the predictors...')
    
    if predictors.empty:
        predictors = 0
        
    import sklearn.preprocessing
    imp = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0) # impute along columns
    predictors_filled = imp.fit_transform(predictors).copy()
    predictors_filled_df = pd.DataFrame(index=predictors.index,columns=predictors.columns,data=predictors_filled)
        
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
    
    from sklearn.metrics import r2_score
    
    print('Creating a model for '+str(site.name))
    
    # split into known/unknown datapoints
    known_x,known_y,unknown_x = split_known_unknown_rows(predictors,site)
    if len(known_y)<5:
        print('Not enough known values for this station!')
        return None
    
    for p in predictors.columns:
        print('   Correlation for '+p+': '+str(predictors[p].corr(site)))
    
    # shuffle rows
    from sklearn.utils import shuffle
    known_x,known_y = shuffle(known_x.copy(),known_y.copy())
    known_y = known_y.ravel()
    
    # split known into test/train
    num_known = len(known_y)
    train_indx = range(0,int(num_known*.75))
    test_indx = range(int(num_known*.75),num_known)
    print('There are '+str(len(train_indx))+' training points and '+str(len(test_indx))+' testing points.')
    
    # linear model
    import sklearn.linear_model
    lin_model = sklearn.linear_model.LinearRegression()
    lin_model.fit(known_x.iloc[train_indx,:], known_y[train_indx])
    lin_model_predicted = lin_model.predict(known_x.iloc[test_indx])
    r2_lin_test = r2_score(known_y[test_indx],lin_model_predicted)
    lin_model_train_predicted = lin_model.predict(known_x.iloc[train_indx])
    r2_lin_train = r2_score(known_y[train_indx],lin_model_train_predicted)

    # neural network
    import sklearn.neural_network
    #HL1_size = int(len(predictors.columns)*)
    hl_size = (20,3) # should probably depend on training data shape
    model = sklearn.neural_network.MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(hl_size),activation='relu')
    
    '''
    # SVM
    import sklearn.svm
    model = sklearn.svm.SVR()
    '''
    
    '''
    # regression tree
    import sklearn.tree
    model = sklearn.tree.DecisionTreeRegressor(max_depth=3)
    '''    

    # fit the model with the training data
    model.fit(known_x.iloc[train_indx,:], known_y[train_indx])
    model_predicted = model.predict(known_x.iloc[test_indx])
    r2_ML_test = r2_score(known_y[test_indx],model_predicted)
    model_train_predicted = model.predict(known_x.iloc[train_indx])
    r2_ML_train = r2_score(known_y[train_indx],model_train_predicted)
    
    # choose which model to use based on testing r2 value
    if r2_ML_test > r2_lin_test:
        model = model
    else:
        print('Using the linear model.')
        model = lin_model
    
    # test the model on the training data now
    #model_known_predicted = model.predict(known_x.iloc[train_indx])
    #r2_known_predicted = r2_score(known_y[train_indx],model_known_predicted)
    
    # target vs predicted
    fig=plt.figure(figsize=(12,6))
    
    ax1 = fig.add_subplot(121)
    ax1.plot(known_y[test_indx],model_predicted,'x',label='Testing points',color=(0,0,.8))
    ax1.plot(known_y[train_indx],model_train_predicted,'.',label='Training points',color='k')
    ax1.plot([0, np.max(known_y)],[0, np.max(known_y)],color='k')
    ax1.set_xlabel('Target')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Machine Learning')
    ax1.legend(loc=4)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(known_y[test_indx],lin_model_predicted,'x',label='Testing points',color=(0,0,.8))
    ax2.plot(known_y[train_indx],lin_model_train_predicted,'.',label='Training points',color='k')
    ax2.plot([0, np.max(known_y)],[0, np.max(known_y)],color='k')
    ax2.set_xlabel('Target')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Linear Model')
    ax2.legend(loc=4)
    
    #plt.title(str(r2_ML_test)+', '+str(r2_ML_train))
    plt.pause(.1)
    plt.show()
    
    #print(str(r2_lin)+', '+str(r2_predicted)+', '+str(r2_known_predicted))
    print('Linear: '+str(r2_lin_test)+' , '+str(r2_lin_train))
    print('ML    : '+str(r2_ML_test)+' , '+str(r2_ML_train))
        
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
            if num_stations > 1:
                r_jk_bar = total_dist/(num_stations-1)
            else:
                r_jk_bar = 0
        
        CW_ijk = 1/float(num_stations) + r_jk_bar/nearby_metadata.loc[station]['Distance']    
        R_ij = (1/nearby_metadata.loc[station]['Distance'] )**2    
        station_weights[station] = R_ij * CW_ijk
        
    nearby_metadata['weight'] = station_weights
    
    return nearby_metadata
    
# re-compute the station weights based on which stations have available data.
# nearby_metadata is used to just take out the "available stations"
def spatial_interp_variable_weights(nearby_data,nearby_metadata):
    
    #print(nearby_metadata)
    
    dates = nearby_data.index
    data = pd.Series(index=dates)
    
    # perform weighted average of stations for this day 
    for date in dates:
        
        #print(date)
        
        # get weights for this day
        available_stations = list()
        for station in nearby_data.columns:
            if pd.notnull(nearby_data.loc[date,station]) and (station in nearby_metadata.index):
                available_stations.append(station)
        #print(available_stations)
        useful_metadata = nearby_metadata.copy().loc[available_stations,:]
        useful_metadata = create_station_weights(useful_metadata)
                
        weights_sum = 0
        values_sum = 0
        for station in useful_metadata.index:
            if pd.notnull(nearby_data.loc[date,station]):
                weights_sum = weights_sum + useful_metadata.loc[station,'weight']
                values_sum = values_sum + nearby_data.loc[date,station]*useful_metadata.loc[station,'weight']

        if weights_sum is not 0: # avoid dividing by zero--if no data for any of them, keep it as NaN
            data[date] = values_sum/weights_sum
        else:
            data[date] = np.nan
            
    return data

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
def plot_station_locs(stations,target_latlon):
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    
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
        
    (x,y) = m(target_latlon[1],target_latlon[0])
    m.plot(x,y,'x',color = 'r',ms=8)
    
    return fig    
    
# do everything to get air quality data
def predict_aq_vals(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data,other_data,ignore_closest=False,return_lots=False):
    
    # this will store the metadata for each station that'll be used
    stations = identify_nearby_stations(latlon,r_max_interp,all_data.copy(),start_date,end_date) # look at the data to find ones close enough
    stations = addon_stationid(stations) # give each an id
    stations = remove_dup_stations(stations) # remove the duplicates

    # get rid of the closest station if you want to use that for validation.
    # also save its reading so you can compare later
    closest = None # name of the closest station to ignore
    if ignore_closest:
        closest = stations.index[0]
        print('Ignoring the closest station: '+closest+', which is at ('+str(stations.loc[closest,'Latitude'])+', '+str(stations.loc[closest,'Longitude'])+').')
        
        closest_obj = aq_station(closest)
        closest_obj.latlon = (stations.loc[closest,'Latitude'],stations.loc[closest,'Longitude'])
        closest_obj.start_date = start_date
        closest_obj.end_date = end_date
        closest_obj.get_station_data(r_max_interp,all_data.copy(),other_data.copy())
        
        # store the "target data" for comparison
        target_data = closest_obj.this_station
        
        # get rid of this and other stations at that same location
        stations = stations[stations['Distance']>0.1]
        
        print('Stations, before weights are computed:')
        print(stations)
        
        # try predicting the values without filling in missing ones with ML
        print('Predicting the values at this station without imputation...')
        results_noML = spatial_interp_variable_weights(closest_obj.nearby_data_df,stations)
        plt.figure()
        plt.plot(results_noML,label='results, no ML')
        plt.plot(target_data,label='target')
        plt.legend()
        plt.show()
    
    # metadata for stations used in the spatial interpolation
    stations = create_station_weights(stations)    
    print('Stations, with weights, used in the spatial interpolation:')
    print(stations)
    
    # plot these stations on a map
    #plot_station_locs(stations,latlon)
    
    print('-----------------------------------------------------------')
    print('Now beginning to fill in the missing data in the above stations.')
    print('-----------------------------------------------------------')
    
    # for each nearby station, fill in missing data
    orig = pd.DataFrame(columns=stations.index.copy())
    composite_data = pd.DataFrame()
    station_obj_list = list()
    for station in stations.index:
    
        station_obj = None
        
        # initialize a station with its name and coordinates
        station_obj = aq_station(station,ignoring=closest_obj.latlon)
        station_obj.latlon = (stations.loc[station,'Latitude'],stations.loc[station,'Longitude'])
        station_obj.start_date = start_date
        station_obj.end_date = end_date
        
        # extract data from neaerby stations in the EPA database
        station_obj.get_station_data(r_max_ML,all_data.copy(),other_data.copy())
        
        # make a copy of this station's original data
        orig.loc[:,station] = station_obj.this_station.copy()
        
        # create and run a model to fill in missing data
        station_obj.create_model()
        station_obj.run_model()
        
        # store the object for this station and add the filled-in data to the composite dataset
        station_obj_list.append(station_obj)        
        composite_data.loc[:,station] = station_obj.composite_data.rename(station).copy()
        
    print('-----------------------------------------------------------')
    print('Done filling in missing data from the nearby stations.')
    print('-----------------------------------------------------------')
        
    # using the composite dataset constructed above, perform the spatial interpolation algorithm
    if composite_data.isnull().values.any():
        print('Recalculating weights for each day...')
        data = spatial_interp_variable_weights(composite_data,stations)
    else:
        print('No NaNs found, so using constant weights...')
        data = spatial_interp(composite_data,stations)   
        
    # plot the predicted, original, and composite data
    final_big_plot(data,orig,composite_data,stations)    
    matrix_val_plot(orig)
    matrix_val_plot(composite_data)    
    
    print('Stations used for the interpolation:')
    print(stations)
    
    # if the closest station was ignored (for validation purposes), plot that
    # known data against the predicted data
    if ignore_closest:
        
        '''
        # both time series
        plt.figure()
        plt.plot(data,label='predicted')
        plt.plot(results_noML,label='predicted, no ML')
        plt.plot(target_data,'.-',label='target')
        plt.legend()
        plt.show()
        '''
        '''
        
        from sklearn.metrics import r2_score
        compare_df = pd.DataFrame()
        compare_df['predicted'] = data.copy()
        compare_df['target'] = target_data.copy()
        compare_df['predicted_noML'] = results_noML.copy()
        compare_df = compare_df[np.isfinite(compare_df['target'])]
        r2 = r2_score(compare_df['predicted'],compare_df['target'])
        
        try:
            # one against the other
            plt.figure()
            plt.scatter(compare_df['target'],compare_df['predicted'],label='with ML')
            plt.scatter(compare_df['target'],compare_df['predicted_noML'],label='no ML')
            plt.plot([0,0],[data.max(),data.max()])
            plt.legend()
            plt.ylabel('Predicted')
            plt.xlabel('Target')
            plt.show()
        except:
            print('error plotting')
        '''
            
        if return_lots == True:
            return data, target_data, results_noML, station_obj_list, composite_data, orig
        else:
            return data, target_data, results_noML
    
    else:
        
        if return_lots==False:
            return data
        else:
            return data, station_obj_list, composite_data, orig