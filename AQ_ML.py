# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 19:14:39 2017

@author: danjr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
This module contains functions for extracting air quality data from the EPA AQS
dataset, imputing missing data, and estimating the air quality with a method
similar to that of Falke and Husar as described here: 
http://capita.wustl.edu/capita/capitareports/mappingairquality/mappingaqi.pdf
'''

# some constants
R_earth =  6371.0 # [km]

def matshow_dates(df,ax):
    '''
    Plot a dataframe as a matrix color plot, using the dates in the index as 
    the x-axis
    '''
    
    import matplotlib.dates as mdates
    xlims = [mdates.date2num(pd.to_datetime(df.index[x])) for x in[0,-1]]
    ax.matshow(df.copy().transpose(),aspect='auto',extent=[xlims[0],xlims[1],len(df.columns),0],origin='upper')
    ax.set_yticklabels(df.columns.values)
    ax.set_yticks([x+0.5 for x in range(0,len(df.columns.values))])
    ax.xaxis.tick_bottom()
    ax.xaxis_date()
    plt.show()
    plt.pause(0.01)
    plt.show()
    return ax

def nn_viz_map(station,ax=None):
    '''
    Visualize a neural network with the nodes plotted over a Basemap.    
    '''
    
    from mpl_toolkits.basemap import Basemap
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    if ax is None:
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
    
    all_nearby_stations = pd.concat([station.nearby_stations,station.other_stations])    
    
    nearby_stations = [n for n in station.gs.columns if n in all_nearby_stations.index]
    
    '''
    Set up the extent of the basemap
    '''
    
    left_lim = all_nearby_stations['Longitude'].min()
    right_lim = all_nearby_stations['Longitude'].max()
    bottom_lim = all_nearby_stations['Latitude'].min()
    top_lim = all_nearby_stations['Latitude'].max()
    
    dlon = right_lim - left_lim
    dlat = top_lim - bottom_lim
    
    left_lim = left_lim - dlon*0.1
    right_lim = right_lim + dlon*0.1
    bottom_lim = bottom_lim - dlat*0.1
    top_lim = top_lim + dlat*0.1
    
    # create a Basemap and add relevant features to the lowest plane of the figure
    m = Basemap(projection='cyl',llcrnrlon=left_lim,llcrnrlat=bottom_lim,urcrnrlon=right_lim,urcrnrlat=top_lim,ax=ax,fix_aspect=True,resolution='h')
    
    ax.add_collection3d(m.drawcoastlines(linewidth=0.5))
    ax.add_collection3d(m.drawcountries(linewidth=0.5))
    ax.add_collection3d(m.drawstates(linewidth=.35))
    ax.add_collection3d(m.drawrivers(linewidth=0.35))
    #ax.add_collection3d(m.drawcounties(linewidth=0.2))
    ax.set_axis_off()
    
    plt.show()
    
    '''
    Plot the three planes. The physical location of the stations will go on the
    bottom; the hidden layer nodes will go on the middle; the output will go on 
    the top.
    '''
    
    (mlons,mlats) = m([left_lim,right_lim],[bottom_lim,top_lim])
    
    alpha= 0.2
    
    # plot the middle plane
    x = [mlons[0],mlons[0],mlons[1],mlons[1]]
    y = [mlats[0],mlats[1],mlats[1],mlats[0]]
    z = [1,1,1,1]
    verts = [zip(x,y,z)]
    ax.add_collection3d(Poly3DCollection(verts,color=(0,0,1,alpha)))
    plt.show()
    
    # plot the top plane
    x = [mlons[0],mlons[0],mlons[1],mlons[1]]
    y = [mlats[0],mlats[1],mlats[1],mlats[0]]
    z = [2,2,2,2]
    verts = [zip(x,y,z)]
    ax.add_collection3d(Poly3DCollection(verts,color=(1,0,0,alpha)))
    plt.show()
    
    # plot the bottom plane
    x = [mlons[0],mlons[0],mlons[1],mlons[1]]
    y = [mlats[0],mlats[1],mlats[1],mlats[0]]
    z = [0,0,0,0]
    verts = [zip(x,y,z)]
    ax.add_collection3d(Poly3DCollection(verts,alpha=0.2,color=(0,1,0,alpha)))
    plt.show()
    
    # plot the back planes
    x = [mlons[0],mlons[0],mlons[1],mlons[1]]
    y = [mlats[1],mlats[1],mlats[1],mlats[1]]
    z = [0,2,2,0]
    verts = [zip(x,y,z)]
    ax.add_collection3d(Poly3DCollection(verts,alpha=0.2,color=(.5,.5,.5,alpha)))
    plt.show()
    x = [mlons[0],mlons[0],mlons[0],mlons[0]]
    y = [mlats[0],mlats[0],mlats[1],mlats[1]]
    z = [0,2,2,0]
    verts = [zip(x,y,z)]
    ax.add_collection3d(Poly3DCollection(verts,alpha=0.2,color=(.5,.5,.5,alpha)))
    plt.show()
    
    '''
    Plot the nodes on their respective planes. For nodes not in the input
    layer, the location is determined by the weighted average of the nodes
    feeeding into it.
    '''
    
    hl_size = station.model.hidden_layer_sizes
    
    
    # plot the input layer
    weights_sum_dict = {}
    weights_max_dict = {} # max. abs. of the weights going into a node
    neuron_loc_dict = {}
    for hl_num in range(hl_size):
        weighted_x = 0
        weighted_y = 0
        weights_sum = 0
        weights_max_dict[hl_num] = 0
        for ix,nearby_station_indx in enumerate(nearby_stations):
            nearby_station = all_nearby_stations.loc[nearby_station_indx]
            (x,y) = m([nearby_station['Longitude']],[nearby_station['Latitude']])
            x = x[0]
            y = y[0]
            
            weight = abs(station.model.coefs_[0][ix,hl_num])
            
            weighted_x = weighted_x + x*weight
            weighted_y = weighted_y + y*weight
            weights_sum= weights_sum + weight
            weights_max_dict[hl_num] = max(abs(weight),weights_max_dict[hl_num])
            
        x = weighted_x / weights_sum
        y = weighted_y / weights_sum
        weights_sum_dict[hl_num] = weights_sum
        
        neuron_loc_dict[hl_num] = (x,y)
        ax.plot([x],[y],'o',color='b',lw=1,zs=1)            
        
    for ix,nearby_station_indx in enumerate(nearby_stations):
        
        nearby_station = all_nearby_stations.loc[nearby_station_indx]
        (x,y) = m([nearby_station['Longitude']],[nearby_station['Latitude']])
        ax.plot(x,y,'o',color='g',lw=1,zs=0)
        
        for hl_num in range(hl_size):
            weight = abs(station.model.coefs_[0][ix,hl_num])            
            ax.plot([x[0],neuron_loc_dict[hl_num][0]],[y[0],neuron_loc_dict[hl_num][1]],zs=[0,1],color='k',lw=float(weight)/weights_max_dict[hl_num]*3,alpha=0.6)
      
    weighted_x = 0
    weighted_y = 0
    weights_sum = 0      
    weights_max = 0
    
    for hl1_num in range(hl_size):
        
        (x,y) = neuron_loc_dict[hl1_num]
        weight = abs(station.model.coefs_[1][hl1_num])
        
        weighted_x = weighted_x + x*weight
        weighted_y = weighted_y + y*weight
        weights_sum= weights_sum + weight
        weights_max = max(weights_max,abs(weight))
        
    final_x = weighted_x / weights_sum
    final_y = weighted_y / weights_sum
    ax.plot([final_x],[final_y],'o',color='r',lw=1,zs=2)
    
    for hl1_num in range(hl_size):
        weight = abs(station.model.coefs_[1][hl1_num])
            
        (x,y) = neuron_loc_dict[hl1_num]
        ax.plot([final_x[0],neuron_loc_dict[hl1_num][0]],[final_y[0],neuron_loc_dict[hl1_num][1]],zs=[2,1],color='k',lw=float(weight/weights_max)*3,alpha=0.6)
            
    # column for the "output" station
    (x_target,y_target) = m([station.latlon[1]],[station.latlon[0]])
    
    ax.plot([x_target[0],final_x[0]],[y_target[0],final_y[0]],zs=[0,2],color='k')
    ax.plot([x_target[0]],[y_target[0]],'x',zs=[0],color='g')
    
    
    plt.show()
    
'''
def nn_viz(model,predictor_names):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)    
    
    layervals = list()
    for n1 in range(len(model.coefs_[0])):
        for n2 in range(len(model.coefs_[0][0])):
            layervals.append(model.coefs_[0][n1][n2])
    abslv = [abs(x) for x in layervals]
    for n1 in range(len(model.coefs_[0])):
        for n2 in range(len(model.coefs_[0][0])):
            g = max(0,model.coefs_[0][n1][n2]/max(layervals))
            r = (model.coefs_[0][n1][n2]<1) * max(0,model.coefs_[0][n1][n2]/min(layervals))
    
            ax.plot([0,1],[float(n1)/(len(model.coefs_[0])-1),float(n2)/(len(model.coefs_[0][0])-1)],lw=abs(model.coefs_[0][n1][n2])/max(abslv)*5,color=(r,g,0))
    
    layervals = list()    
    for n2 in range(len(model.coefs_[0][0])):
        for n3 in range(len(model.coefs_[1][0])):
            layervals.append(model.coefs_[1][n2][n3])        
    abslv = [abs(x) for x in layervals]
    for n2 in range(len(model.coefs_[0][0])):
        for n3 in range(len(model.coefs_[1][0])):
            g = max(0,model.coefs_[1][n2][n3]/max(layervals))
            r = (model.coefs_[1][n2][n3]<1) * max(0,model.coefs_[1][n2][n3]/min(layervals))
            ax.plot([1,2],[float(n2)/(len(model.coefs_[0][0])-1),float(n3)/(len(model.coefs_[1][0])-1)],lw=abs(model.coefs_[1][n2][n3])/max(abslv)*5,color=(r,g,0))
      
    layervals = list()    
    for n3 in range(len(model.coefs_[1][0])):
        layervals.append(model.coefs_[2][n3])        
    abslv = [abs(x) for x in layervals]
    for n3 in range(len(model.coefs_[1][0])):
        g = max(0,model.coefs_[2][n3]/max(layervals))
        r = (model.coefs_[2][n3]<1) * max(0,model.coefs_[2][n3]/min(layervals))
        ax.plot([2,3],[float(n3)/(len(model.coefs_[1][0])-1),0.5],lw=abs(model.coefs_[2][n3])/max(abslv)*5,color=(r,g,0))
        
        
    for n in range(len(model.coefs_[0])):
        ax.plot(0,float(n)/(len(model.coefs_[0])-1),'o',color='k')
    plt.show()
    
    for n in range(len(model.coefs_[0][0])):
        ax.plot(1,float(n)/(len(model.coefs_[0][0])-1),'o',color='k')
    plt.show()
    
    for n in range(len(model.coefs_[1][0])):
        ax.plot(2,float(n)/(len(model.coefs_[1][0])-1),'o',color='k')
    plt.show()
    
    ax.plot(3,0.5,'o',color='k')
    
    ax.set_yticks(np.linspace(0,1,len(predictor_names)))
    ax.set_yticklabels(predictor_names)
        
    # plot the inputs
    #for x in 
'''
    
def compare_dfs_plot(composite,original):
    '''
    After the imputation procedure is run, compare the original and composite
    datasets by plotting each as matrices.
    '''
    fig = plt.figure(figsize=(7,9))
    
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212,sharex=ax1)
    
    matshow_dates(original,ax1)
    matshow_dates(composite,ax2)
    

class aq_station:
    '''
    Class for an air quality station. Includes methods to get and manipulate 
    the station's data and to impute the missing data based on data from other
    stations nearby.
    '''
    
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
        
        start = df['Date Local'].min()
        end = df['Date Local'].max()
        
        # get data of interest
        self.nearby_stations = identify_nearby_stations(self.latlon,r_max,df,start,end)
        self.nearby_stations = addon_stationid(self.nearby_stations)
        self.nearby_stations = remove_dup_stations(self.nearby_stations,ignore_closest=False)
        if self.ignoring is not None:
            print('   Removing stations with latitude '+str(self.ignoring[0]))
            self.nearby_stations = self.nearby_stations[self.nearby_stations['Latitude']!=self.ignoring[0]].copy()        
        self.nearby_data_df = extract_nearby_values(self.nearby_stations,df,start,end)
        
        # separate this station's data from the "nearby" dataframe
        if self.station_id in self.nearby_data_df.columns:
            self.this_station = pd.Series(self.nearby_data_df[self.station_id]).copy()
            self.nearby_data_df = self.nearby_data_df.drop(self.station_id, axis=1)
        else:
            self.this_station = pd.Series()
            print('No data for this station!')        
        
        # get the data for the auxillary stations
        self.other_stations = identify_nearby_stations(self.latlon,r_max,other_data,start,end)
        self.other_stations = addon_stationid(self.other_stations)
        self.other_stations = remove_dup_stations(self.other_stations,ignore_closest=False)
        if self.ignoring is not None:
            print('   Removing stations with latitude '+str(self.ignoring[0]))
            self.other_stations = self.other_stations[self.other_stations['Latitude']!=self.ignoring[0]].copy()        
        self.other_data_df = extract_nearby_values(self.other_stations,other_data,start,end)
        
    def plot_matrix_station(self):
        
        #print('Making plot for this station!')
        
        #import matplotlib.dates as mdates
        #import datetime as dt
                
        fig = plt.figure(figsize=(12,6))
        if self.gs.empty:
            print('   ...good sites are empty.')
            return fig

        ax1 = fig.add_subplot(211)
        
        ax2 = fig.add_subplot(212,sharex=ax1)
        matshow_dates(self.gs,ax2)
        
        ax1.plot(self.composite_data,'.-',color='red',label='Imputed data')
        ax1.plot(self.this_station,'.-',lw=2,color='k',label='Original data')
        ax1.set_ylabel(self.this_station.name)
        ax1.set_title('r2 = '+str(self.model_r2))
        #ax1.set_ylabel('Output station')
        ax1.legend()
        
        roll_known = self.this_station.rolling(window=14,center=True,min_periods=0)
        roll_pred = self.composite_data.rolling(window=14,center=True,min_periods=0)
                
        #from sklearn.metrics import r2_score
        
        rolling_corr = roll_known.mean() / roll_pred.mean()
        
        ax3 = ax1.twinx()
        ax3.plot(rolling_corr,color='c')
        
        plt.show()
        plt.pause(.1)
        
        return fig        
        
    def create_model(self):
        
        # determine which features should be used for this model
        days = pd.Series(index=self.nearby_data_df.index,data=(self.nearby_data_df.index-self.nearby_data_df.index[0])/pd.Timedelta('1D'))
        days = days.rename('days')
        self.gs,self.bs = feature_selection_rfe(pd.concat([self.nearby_data_df,self.other_data_df],axis=1),self.this_station,self.start_date,self.end_date) # nearby_data_df does NOT include the station to predict
            
        if self.gs.empty:
            print('No good sites found to make this model. No model being created...')
            self.model = None
            return
            
        # fill missing predictors and normalize the inputs
        self.gs = fill_missing_predictors(self.gs)
        
        # create a model
        self.gs_columns = tuple(self.gs.columns)
        self.model,self.model_r2 = create_model_for_site(self.gs,self.this_station)
        
        import sklearn.neural_network
        if isinstance(self.model,sklearn.neural_network.MLPRegressor):
            nn_viz_map(self)
        
    def run_model(self):
        gs_use = self.gs.copy()[(self.gs.index>=self.start_date)&(self.gs.index<=self.end_date)]
        this_station_use = self.this_station.copy()[(self.this_station.index>=self.start_date)&(self.this_station.index<=self.end_date)]
        gs_use = gs_use[list(self.gs_columns)]
        self.composite_data = fill_with_model(gs_use,this_station_use,self.model)
        
        # plot the features and the value to predict
        self.plot_matrix_station()
        
        self.composite_data[pd.notnull(self.this_station)] = self.this_station

def extract_raw_data(start_date,end_date,param_code=81102):
    
    folder = 'C:\Users\danjr\Documents\ML\Air Quality\data\\'
    #folder = 'C:\Users\druth\Documents\epa_data\\'
    
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
    
def identify_sampling_rate(series):
    '''
    Given a series of reported data from a station, infer if it's on the 1, 3,
    6, or 12 day reporting schedule.
    '''
    
    is_nan = pd.isnull(series)        
    good_dates = series.index[is_nan==False]
    early = pd.to_datetime(good_dates[1:])
    later = pd.to_datetime(good_dates[0:-1])
    
    diff_data = early-later
    diff_period = pd.Series(index=good_dates[0:-1],data=diff_data)
    estimated_rate = pd.Timedelta(np.median(diff_period))
    
    return estimated_rate
    
def identify_nearby_stations(latlon,r_max,df,start_date,end_date,ignore_closest=False):
    '''
    Get the metadata for stations within r_max of latlon in df.
    '''
    
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
    
    #param_stations = addon_stationid(param_stations)
    #param_stations = remove_dup_stations(param_stations,ignore_closest=False)

    return param_stations    
    
def addon_stationid(df):
    '''
    Given a dataframe of stations, add on a column with the "station id" string
    (identifying its type, state, county, and POC).
    '''
    u_col = pd.Series(index=df.index,data='_')    
    station_ids = df['Parameter Code'].map(str)+u_col+df['State Code'].map(str)+u_col+df['County Code'].map(str)+u_col+df['Site Number'].map(str)+u_col+df['POC'].map(str)

    df['station_ids'] = station_ids    
    
    return df
    
def remove_dup_stations(param_stations,ignore_closest=False):
    '''
    Remove duplicate rows (corresponding to the same station).
    
    Also, get rid of all stations within 0.5 km of the "target" location, if
    desired (this is used in validation).
    '''
    
    # make the IDS the index, and get rid of duplicates
    param_stations = param_stations.set_index('station_ids')
    param_stations = param_stations[~param_stations.index.duplicated(keep='first')]
    
    if ignore_closest:
        param_stations = param_stations[param_stations['Distance']>0.5]
    
    return param_stations
    
# pick out the values from stations nearby.
# this is called separately for the main and auxilliary dataframes
def extract_nearby_values(stations,all_data,start_date,end_date):
    '''
    Given some stations in the input stations and a df of lots of stations'
    readings in all_data, extract the readings from the desired stations.
    '''
    
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

def feature_selection_rfe(df,this_station,start_date,end_date,stations_to_keep=None):
    '''
    Determine which nearby stations to use as predictors in filling in missing
    data from this_station.
    '''
        
    # get the known values of the station to predict during the prediction period
    this_station_duringpred = this_station[(this_station.index>=start_date)&(this_station.index<=end_date)]
    
    missing_days_period = this_station_duringpred.index[pd.isnull(this_station_duringpred)]
    known_days_period = this_station_duringpred.index[pd.notnull(this_station_duringpred)]
    missing_days_all = this_station.index[pd.isnull(this_station)]
    known_days_all = this_station.index[pd.notnull(this_station)]
    
    #print('There are '+str(len(missing_days))+' missing days out of '+str(len(this_station_duringpred))+' total days for this station.')
    
    print('During the prediction period, this station is missing '+str(len(missing_days_period))+' out of '+str(len(this_station_duringpred))+' total days.')
    print('During the whole period, this station is missing '+str(len(missing_days_all))+' out of '+str(len(this_station))+' total days.')
    
    from sklearn.feature_selection import RFE
    #from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    matshow_dates(df,ax)
    ax.set_title('Stations considered for a model for '+str(this_station.name))
    
    cols_to_consider = list()
    
    # look at each column (data for a given station) and see if it's got enough datapoints
    for column in df:
        
        col_vals = df[column]
        
        # for the period of prediction, determine when this potential predictor is/isn't missing data
        col_while_missing_period = col_vals[missing_days_period]
        #col_while_known_period = col_vals[known_days_period]
        #col_while_missing_all = col_vals[missing_days_all]
        col_while_known_all = col_vals[known_days_all]
        
        # now that the portion missing is calculated, fill in the missing values
        col_vals.loc[pd.isnull(col_vals)] = col_vals[pd.notnull(col_vals)].mean()
        
        # identify the portion of values missing from this predictor station (col) while
        # we also are/are not missing values from the station to predict (this_station).
        num_missing_while_missing_period = len(col_while_missing_period[pd.isnull(col_while_missing_period)==True]) # missing days from (column when this_station is missing)
        if len(missing_days_period)==0:
            portion_missing_while_missing_period = 0
        else:
            portion_missing_while_missing_period = float(num_missing_while_missing_period)/float(len(missing_days_period))
        num_missing_while_known_all = len(col_while_known_all[pd.isnull(col_while_known_all)==True]) # missing days from (column when this_station is missing)
        if len(known_days_all)==0:
            portion_missing_while_known_all = 1
        else:
            portion_missing_while_known_all = float(num_missing_while_known_all)/float(len(known_days_all))
        
        # consider when/how much data is missing and decide whether or not to 
        # use this station as a predictor
        consider_col = (portion_missing_while_missing_period < 0.2) & (portion_missing_while_known_all < 0.4)       
        
        if consider_col==True:
            #print([portion_missing_while_missing,portion_missing_while_known])
            cols_to_consider.append(column)
        
        df[column] = col_vals

    known_x,known_y,unknown_x = split_known_unknown_rows(df[cols_to_consider],this_station)
    
    # choose how many stations to keep based on how many samples there will be to train on
    if stations_to_keep is None:
        stations_to_keep = min(15,max(1,int(len(known_y)/20)))
    
    model = DecisionTreeRegressor(max_depth=5)
    n_features = max(3,min(int(float(len(known_days_all))/float(50)),10))
    print('Picking out '+str(n_features)+' predictor stations.')
    rfe = RFE(model,n_features)
    
    fit = rfe.fit(known_x,known_y)
    #print(fit)
    
    feature_is_used = fit.support_
    cols_to_keep = list()
    for ix in range(len(cols_to_consider)):
        if feature_is_used[ix] == True:
            cols_to_keep.append(cols_to_consider[ix])
    #cols_to_keep = cols_to_consider[feature_is_used==True]
    #cols_to_keep.append('days')
    print(cols_to_keep)
    good_stations_filtered = df.loc[:,cols_to_keep]
    
    return good_stations_filtered, None
    
# fill in missing predictor values, keeping it as a df
def fill_missing_predictors(predictors):    
    
    print('Filling in the missing values from the predictors...')
    
    if predictors.empty:
        predictors = 0
        
    import sklearn.preprocessing
    imp = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0) # impute along columns
    predictors_filled = imp.fit_transform(predictors.copy())
    predictors_filled_df = pd.DataFrame(index=predictors.index,columns=predictors.columns,data=predictors_filled)
    
    
    from scipy.stats import zscore
    predictors_filled_df = predictors_filled_df.apply(zscore)

    return predictors_filled_df
    
# based on which values for the site are available, split up predictors/known
# into known and unknown
def split_known_unknown_rows(predictors,site):
    
    known_x = predictors[pd.notnull(site)]
    known_y = site[pd.notnull(site)]
    unknown_x = predictors[pd.isnull(site)]
    
    return known_x,known_y,unknown_x
    
def create_model_for_site(predictors,site):
    '''
    Create a linear model or neural network to fill in the missing data for a
    site.
    '''
    
    from sklearn.metrics import r2_score    
    
    print('Creating a model for '+str(site.name))    
    
    # split into known/unknown datapoints
    known_x,known_y,unknown_x = split_known_unknown_rows(predictors,site)
    if len(known_y)<20:
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
    hl1_size = max(2,min(max(2,int(num_known/180)),len(predictors.columns)-2))
    #hl_size = (hl1_size,min(4,max(2,hl1_size/2))) # should probably depend on training data shape
    hl_size = hl1_size
    #hl_size = (hl1_size,1) # should probably depend on training data shape
    print(str(hl_size)+' hidden layer nodes.')
    model = sklearn.neural_network.MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(hl_size),activation='relu')
    
    
    '''
    # SVM
    import sklearn.svm
    model = sklearn.svm.SVR()
    '''
    
    '''
    # regression tree
    import sklearn.tree
    model = sklearn.tree.DecisionTreeRegressor(max_depth=5)
    '''
        

    # fit the model with the training data
    model.fit(known_x.iloc[train_indx,:], known_y[train_indx])
    #nn_viz(model,predictors.columns)
    model_predicted = model.predict(known_x.iloc[test_indx])
    r2_ML_test = r2_score(known_y[test_indx],model_predicted)
    model_train_predicted = model.predict(known_x.iloc[train_indx])
    r2_ML_train = r2_score(known_y[train_indx],model_train_predicted)
    
    # choose which model to use based on testing r2 value
    if (r2_ML_test < .3) and (r2_lin_test < .3):
        print('Both r2s < 0.3, not creating a model.')
        return None, None
        
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
    ax1.set_title('Machine Learning, r2 = '+str(r2_ML_test))
    ax1.legend(loc=4)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(known_y[test_indx],lin_model_predicted,'x',label='Testing points',color=(0,0,.8))
    ax2.plot(known_y[train_indx],lin_model_train_predicted,'.',label='Training points',color='k')
    ax2.plot([0, np.max(known_y)],[0, np.max(known_y)],color='k')
    ax2.set_xlabel('Target')
    ax2.set_ylabel('Predicted')
    ax2.set_title('Linear Model, r2 = '+str(r2_lin_test))
    ax2.legend(loc=4)
    
    #plt.title(str(r2_ML_test)+', '+str(r2_ML_train))
    plt.pause(.1)
    plt.show()
    
    #print(str(r2_lin)+', '+str(r2_predicted)+', '+str(r2_known_predicted))
    print('Linear: '+str(r2_lin_test)+' , '+str(r2_lin_train))
    print('ML    : '+str(r2_ML_test)+' , '+str(r2_ML_train))
        
    return model, (r2_ML_test,r2_lin_test)

# use the model to fill the missing data, returning a "composite" series
def fill_with_model(predictors,site,model):
    
    print predictors.columns
    
    if model is None:
        return site
    
    # split known/unknown, simulate
    known_x,known_y,unknown_x = split_known_unknown_rows(predictors,site)    
    
    if len(unknown_x)==0:
        return known_y
    print unknown_x.columns
    predicted_y = model.predict(predictors)
    predicted_y = pd.Series(index=predictors.index,data=predicted_y)
    
    '''
    # replace missing with the simulated, returning the composite
    composite_series = site.copy() # start with site data
    composite_series[predicted_y.index] = predicted_y.copy()
    composite_series.loc[composite_series<0] = 0 # just in case
    '''
    composite_series = predicted_y
    
    return composite_series
    
# once nearby stations have been picked, add on a column of their weights (for
# the spatial interpolation algorithm)
def create_station_weights(nearby_metadata,max_stations=10):
    
    # determine the weighting for the stations
    station_weights = pd.Series(index=nearby_metadata.index)
    #nearby_metadata = nearby_metadata.ix[0:min(max_stations,len(nearby_metadata)),:]
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
def spatial_interp_variable_weights(nearby_data,nearby_metadata,max_stations=10):
    
    #print(nearby_metadata)
    
    dates = nearby_data.index
    data = pd.Series(index=dates)
    
    # perform weighted average of stations for this day 
    for date in dates:
        
        print date
        
        # get weights for this day
        this_days_readings = nearby_data.loc[date,:]
        #print(this_days_readings)
        this_days_notnulls = this_days_readings[pd.notnull(this_days_readings)]
        #print(this_days_notnulls)
        available_stations = list(this_days_notnulls.index)
        #print(available_stations)
        #available_stations = available_stations[0:min(len(available_stations),max_stations)]
        #print(available_stations)
        '''
        for station in nearby_data.columns:
            if len(available_stations) < max_stations:
                if pd.notnull(nearby_data.loc[date,station]) and (station in nearby_metadata.index):
                    available_stations.append(station)
        '''
        useful_metadata = nearby_metadata.copy().loc[available_stations,:]
        #print(useful_metadata)
        useful_metadata = useful_metadata.iloc[0:min(len(available_stations),max_stations)]
        useful_metadata = create_station_weights(useful_metadata,max_stations=max_stations)
                
        weights_sum = 0
        values_sum = 0
        for station in useful_metadata.index:
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

def create_composite_dataset(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data,other_data,ignore_closest=False):
    '''
    Create a composite dataset containing the measured and simulated air
    quality values at the stations around latlon. This composite dataset can be
    used to predict the air quality at latlon.
    '''
    
    # Look at the data to determine which stations are within r_max_interp of
    # the location of interest, latlon.
    stations = identify_nearby_stations(latlon,r_max_interp,all_data.copy(),start_date,end_date)
    stations = addon_stationid(stations) # give each an id
    stations = remove_dup_stations(stations) # remove the duplicates

    # get rid of the closest station if you want to use that for validation.
    # also save its reading so you can compare later
    closest = None # name of the closest station to ignore
    if ignore_closest:
        closest = stations.index[0]
        print('Ignoring the closest station: '+closest+', which is at ('+str(stations.loc[closest,'Latitude'])+', '+str(stations.loc[closest,'Longitude'])+').')
        
        # ensure that the coords of this "closest" station are actually the
        # coords being simulated
        if (abs(stations.loc[closest,'Latitude'] - latlon[0]) > 0.001) or (abs(stations.loc[closest,'Longitude'] - latlon[1]) > 0.001):
            raise(ValueError('Input latlon and closest station coordinates do not match!'))
        
        closest_obj = aq_station(closest)
        closest_obj.latlon = (stations.loc[closest,'Latitude'],stations.loc[closest,'Longitude'])
        closest_obj.start_date = start_date
        closest_obj.end_date = end_date
        closest_obj.get_station_data(r_max_interp,all_data.copy(),other_data.copy())
        
        # store the "target data" for comparison
        target_data = closest_obj.this_station
        
        # get rid of this and other stations at that same location
        stations = stations[stations['Distance']>0.1]
        
        '''
        print('Stations, before weights are computed:')
        print(stations)
        
        # try predicting the values without filling in missing ones with ML
        print('Predicting the values at this station without imputation...')
        print(closest_obj.nearby_data_df)
        results_noML = spatial_interp_variable_weights(closest_obj.nearby_data_df,stations,max_stations=10)
        plt.figure()
        plt.plot(results_noML,label='results, no ML')
        plt.plot(target_data,label='target')
        plt.legend()
        plt.show()
        '''
        
    else:
        closest_obj = None
    
    stations = stations.ix[0:min(8,len(stations)),:]
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
        if closest_obj is not None:
            station_obj = aq_station(station,ignoring=closest_obj.latlon)
        else:
            station_obj = aq_station(station,ignoring=None)
        station_obj.latlon = (stations.loc[station,'Latitude'],stations.loc[station,'Longitude'])
        station_obj.start_date = start_date
        station_obj.end_date = end_date
        
        # extract data from nearby stations in the EPA database
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
    
    compare_dfs_plot(composite_data,orig)
    
    if ignore_closest==True:
        return composite_data, orig, stations, station_obj_list, target_data
    
    return composite_data, orig, stations, station_obj_list
    
# do everything to get air quality data
def predict_aq_vals(nearby_data,stations):
    '''
    Given nearby data (which could be the original or composite dataset) and 
    weights associated with each station in the input stations, interpolate to
    get the air quality at a point.
    '''
            
    # using the composite dataset constructed above, perform the spatial interpolation algorithm
    if nearby_data.isnull().values.any():
        print('Recalculating weights for each day...')
        data = spatial_interp_variable_weights(nearby_data,stations)
    else:
        print('No NaNs found, so using constant weights with...')
        print(nearby_data.columns)
        print(stations.index)
        data = spatial_interp(nearby_data,stations)   
    
    print('Stations used for the interpolation:')
    print(stations)

    return data