from __future__ import division
import pandas as pd

class aq_data:
    def __init__(self):
        self.param = ()
        self.daily_average = pd.Series()
        self.monitor_vals= pd.DataFrame()
        self.monitor_info = list()
        self.lat_lon = ()
        self.r_max = ()


R_earth =  6371.0

def nearby_aq(latlong,num_stations,r_max,start_date,end_date,viz=False,param_code = 81102):
    
    
    import pandas as pd
    import numpy as np
    #import soilData as sd
    
    # csv file with description of every EPA monitoring station
    site_ex = pd.read_csv('aqs_monitors.csv')
    
    #param_code = 81102 # EPA code for which parameter to look at (eg 81102 is for PM10 mass)
    
    # distance in km between two lat/lon coordinates
    def lat_lon_dist(point1,point2):
        
        # http://andrew.hedges.name/experiments/haversine/
        
        R_earth =  6371.0 # [km]
    
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
    
    #my_lat = sd.lat_longs[site_name][0]
    #my_lon = sd.lat_longs[site_name][1]
    my_lat = latlong[0]
    my_lon = latlong[1]
    print(my_lat)
    site_ex_paramfiltered = site_ex.ix[site_ex['Parameter Code']==param_code,:]
    
    
    # when each siite was opened/closed
    established_date = pd.to_datetime(site_ex_paramfiltered['First Year of Data'])
    closed_date = pd.to_datetime(site_ex_paramfiltered['Last Sample Date'])
    
    # which sites were open during the range of interest
    good_sites_time = (established_date <= (start_date)) & ((closed_date >= (end_date)) | (closed_date.isnull()==1))
    site_ex_timefiltered = site_ex_paramfiltered.ix[good_sites_time,:]
    site_ex_timefiltered = site_ex_paramfiltered

    # distance to each EPA station
    d = lat_lon_dist([site_ex_timefiltered['Latitude'],site_ex_timefiltered['Longitude']],[my_lat,my_lon])
    site_ex_timefiltered.loc[:,'Distance'] = d
    site_ex_sorted = site_ex_timefiltered.sort_values(['Distance'],ascending=True)
    site_ex_sorted['weight']=np.nan
    print(site_ex_sorted)

    all_data = pd.read_csv('daily_81102_allYears.csv',usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean'])
    
    # compute weighted average separately for each day
    station_readings = pd.DataFrame(index=pd.date_range(start_date,end_date))
    station_weights = pd.DataFrame(index=pd.date_range(start_date,end_date))
    daily_average = pd.Series()
    daily_monitor_info = {} # weights, etc are stored in a separate entry in this for each date
    station_list = pd.DataFrame()
    

    # see which stations have data for this day, and store their info
    found_enough_stations = 0
    num_stations = 0
    i = 0 # index of site descriptions
    j = 0 # counter for how many stations are found for this day
    #monitor_info = list([])
    last_id = ''
    while found_enough_stations == 0: # look through stations
        
        # this info about the site will be stored if it is determined that the site is worth including in the weighted average
        a = {'State Code':int(site_ex_sorted.iloc[i]['State Code']),
             'County Code':int(site_ex_sorted.iloc[i]['County Code']),
             'Site Num':int(site_ex_sorted.iloc[i]['Site Number']),
             'Weight':site_ex_sorted.iloc[i]['weight'], 'Distance':site_ex_sorted.iloc[i]['Distance'],
             'Latitude':site_ex_sorted.iloc[i]['Latitude'],'Longitude':site_ex_sorted.iloc[i]['Longitude'],
             'list_loc':i,
             'station_id':str(site_ex_sorted.iloc[i]['State Code'])+'_'+str(site_ex_sorted.iloc[i]['County Code'])+'_'+str(site_ex_sorted.iloc[i]['Site Number'])}

        this_id = a['station_id']
        if (i > 0) and (this_id == last_id):
            is_duplicate = 1
        else:
            is_duplicate = 0
        last_id = this_id # reset it for the next station
        
        # if this station might be used, get the time series data from it
        if is_duplicate == 0:
            good_points = (all_data['County Code'] == a['County Code']) & (all_data['State Code'] == a['State Code']) & (all_data['Site Num'] == a['Site Num'])
            good_data = all_data.ix[good_points]
            good_data.loc[:,'Date Local'] = pd.to_datetime(good_data['Date Local'])
            good_data = good_data.set_index('Date Local')
        
        # get all daily readings from this site
        for datetime in pd.date_range(start_date,end_date):
        
            day = datetime
            
            #print('Working on '+str(day))                

            # if there is a reading for this day, include it
            if ((good_data.index == day).any()) & (len(good_data) != 0) & (a['Distance'] <= r_max) & (is_duplicate==0):               
                
                #print(station_readings)
                #print(day)
                
                # get reading and store it in the DF
                this_days_reading = good_data.loc[day,'Arithmetic Mean']
                #print(this_days_reading)
                #print(day)
                #print(a)
                station_readings.loc[day,a['station_id']] = np.mean(this_days_reading) # sometimes there's two values for a day...
                station_readings.sort_index(inplace=True)

                # just in case this is the first time the station is being used, record its location info
                station_list.loc[a['station_id'],'Latitude'] = a['Latitude']
                station_list.loc[a['station_id'],'Longitude'] = a['Longitude']
                station_list.loc[a['station_id'],'Distance'] = a['Distance']

                print('Using a reading from '+a['station_id']+' ('+str(a['Distance'])+' km away) on '+str(day)+': '+str(this_days_reading))
                
            else:
                1==1
                #print('No reading from/station is too far/duplicate '+a['station_id']+' ('+str(a['Distance'])+' km away) on '+str(day))

            # should it stop looking for more stations
        if (a['Distance'] > r_max):
                found_enough_stations = 1
                #daily_monitor_info[day] = monitor_info
                
        i = i+1 # index in site_ex_sorted that we're looking at
    
    
    
    # put data into the air quality object that'll be returned
    aq_obj = aq_data()
    aq_obj.lat_lon = [my_lat,my_lon]
    aq_obj.param = param_code
    aq_obj.r_max = r_max
    aq_obj.station_readings = station_readings
    aq_obj.station_list = station_list
        
    return aq_obj
    
def nearby_plot(aq_obj):
    
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
    my_lon = aq_obj.lat_lon[1]
    my_lat = aq_obj.lat_lon[0]
    r_max = aq_obj.r_max
    param_code = aq_obj.param
    #site_name = aq_obj.site_name
    
    num_stations = len(aq_obj.station_list)
    
    # create colors to correspond to each of the stations
    RGB_tuples = [(x/num_stations, (1-x/num_stations),.5) for x in range(0,num_stations)]
    print(RGB_tuples)
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
    
    # plot each EPA site on the map
    for i in range(0,len(aq_obj.station_list)):
        
        plt.subplot(1,2,1)
        (x,y) = m([aq_obj.station_list.iloc[i]['Longitude'],my_lon],[aq_obj.station_list.iloc[i]['Latitude'],my_lat])
        m.plot(x,y,color = RGB_tuples[i])
        
        
        (x,y) = m(aq_obj.station_list.iloc[i]['Longitude'],aq_obj.station_list.iloc[i]['Latitude'])
        #m.plot(x,y,'o',color = RGB_tuples[i])        
        print(x)
        plt.text(x,y,str(i))
        
    # finish the map plot by putting the soiling station loc and radius on there
    plt.subplot(1,2,1)
    (x,y) = m(my_lon,my_lat)
    m.plot(x,y,'kx',markersize=20,lw=3)
    equi(m, my_lon, my_lat, r_max,color='k')        
    plt.title('Found '+str(num_stations)+' acceptable EPA '+str(param_code)+' stations within '+str(r_max)+' km')        
    plt.show()
    
    # readings for all the stations
#    ax=plt.subplot(2,3,2)
#    print(aq_obj.station_readings)
#    for station in aq_obj.station_readings.columns.values:
#        ax.plot(aq_obj.station_readings[station],'x-',label=station)
#    ax.set_ylabel('Station Reading')
#    plt.show()    
    
    # plot each station reading profile
    for station in aq_obj.station_readings.columns.values:
        plt.subplot(2,2,4)
        plt.plot(aq_obj.station_readings[station],'x-',label=station,color = color_dict[station])
    plt.show()    


    
    return fig
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# this function will determine which EPA stations to use, get their data, and compute the averages
# returns the averages plus info about how it made the calculation
def read_airQuality(latlong,num_stations,r_max,start_date,end_date,viz=False,param_code = 81102,skip_best=False):
    
    
    import pandas as pd
    import numpy as np
    #import soilData as sd
    
    # csv file with description of every EPA monitoring station
    site_ex = pd.read_csv('aqs_monitors.csv')
    
    #param_code = 81102 # EPA code for which parameter to look at (eg 81102 is for PM10 mass)
    
    # distance in km between two lat/lon coordinates
    def lat_lon_dist(point1,point2):
        
        # http://andrew.hedges.name/experiments/haversine/
        
        R_earth =  6371.0 # [km]
    
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
    
    #my_lat = sd.lat_longs[site_name][0]
    #my_lon = sd.lat_longs[site_name][1]
    my_lat = latlong[0]
    my_lon = latlong[1]
    print(my_lat)
    site_ex_paramfiltered = site_ex.ix[site_ex['Parameter Code']==param_code,:]
    
    
    # when each siite was opened/closed
    established_date = pd.to_datetime(site_ex_paramfiltered['First Year of Data'])
    closed_date = pd.to_datetime(site_ex_paramfiltered['Last Sample Date'])
    
    # which sites were open during the range of interest
    good_sites_time = (established_date <= (start_date)) & ((closed_date >= (end_date)) | (closed_date.isnull()==1))
    site_ex_timefiltered = site_ex_paramfiltered.ix[good_sites_time,:]
    site_ex_timefiltered = site_ex_paramfiltered

    # distance to each EPA station
    d = lat_lon_dist([site_ex_timefiltered['Latitude'],site_ex_timefiltered['Longitude']],[my_lat,my_lon])
    site_ex_timefiltered.loc[:,'Distance'] = d
    site_ex_sorted = site_ex_timefiltered.sort_values(['Distance'],ascending=True)
    site_ex_sorted['weight']=np.nan
    print(site_ex_sorted)

    all_data = pd.read_csv('daily_81102_allYears.csv',usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean'])
    
    # compute weighted average separately for each day
    station_readings = pd.DataFrame(index=pd.date_range(start_date,end_date))
    station_weights = pd.DataFrame(index=pd.date_range(start_date,end_date))
    daily_average = pd.Series()
    daily_monitor_info = {} # weights, etc are stored in a separate entry in this for each date
    station_list = pd.DataFrame()
    

    # see which stations have data for this day, and store their info
    found_enough_stations = 0
    num_stations = 0
    i = 0 # index of site descriptions
    j = 0 # counter for how many stations are found for this day
    #monitor_info = list([])
    last_id = ''
    while found_enough_stations == 0: # look through stations
        
        # this info about the site will be stored if it is determined that the site is worth including in the weighted average
        a = {'State Code':int(site_ex_sorted.iloc[i]['State Code']),
             'County Code':int(site_ex_sorted.iloc[i]['County Code']),
             'Site Num':int(site_ex_sorted.iloc[i]['Site Number']),
             'Weight':site_ex_sorted.iloc[i]['weight'], 'Distance':site_ex_sorted.iloc[i]['Distance'],
             'Latitude':site_ex_sorted.iloc[i]['Latitude'],'Longitude':site_ex_sorted.iloc[i]['Longitude'],
             'list_loc':i,
             'station_id':str(site_ex_sorted.iloc[i]['State Code'])+'_'+str(site_ex_sorted.iloc[i]['County Code'])+'_'+str(site_ex_sorted.iloc[i]['Site Number'])}

        this_id = a['station_id']
        if (i > 0) and (this_id == last_id):
            is_duplicate = 1
        else:
            is_duplicate = 0
        last_id = this_id # reset it for the next station
        
        # if this station might be used, get the time series data from it
        if is_duplicate == 0:
            good_points = (all_data['County Code'] == a['County Code']) & (all_data['State Code'] == a['State Code']) & (all_data['Site Num'] == a['Site Num'])
            good_data = all_data.ix[good_points]
            good_data.loc[:,'Date Local'] = pd.to_datetime(good_data['Date Local'])
            good_data = good_data.set_index('Date Local')
        
        # get all daily readings from this site
        for datetime in pd.date_range(start_date,end_date):
        
            day = datetime
            
            #print('Working on '+str(day))                

            # if there is a reading for this day, include it
            if ((good_data.index == day).any()) & (len(good_data) != 0) & (a['Distance'] <= r_max) & (is_duplicate==0):               
                
                #print(station_readings)
                #print(day)
                
                # get reading and store it in the DF
                this_days_reading = good_data.loc[day,'Arithmetic Mean']
                #print(this_days_reading)
                #print(day)
                #print(a)
                station_readings.loc[day,a['station_id']] = np.mean(this_days_reading) # sometimes there's two values for a day...
                station_readings.sort_index(inplace=True)

                # just in case this is the first time the station is being used, record its location info
                station_list.loc[a['station_id'],'Latitude'] = a['Latitude']
                station_list.loc[a['station_id'],'Longitude'] = a['Longitude']
                station_list.loc[a['station_id'],'Distance'] = a['Distance']

                print('Using a reading from '+a['station_id']+' ('+str(a['Distance'])+' km away) on '+str(day)+': '+str(this_days_reading))
                
            else:
                1==1
                #print('No reading from/station is too far/duplicate '+a['station_id']+' ('+str(a['Distance'])+' km away) on '+str(day))

            # should it stop looking for more stations
        if (a['Distance'] > r_max):
                found_enough_stations = 1
                #daily_monitor_info[day] = monitor_info
                
        i = i+1 # index in site_ex_sorted that we're looking at
    
    # compute weights for stations used each day
    for datetime in pd.date_range(start_date,end_date):
        
        day = datetime.date()
        print('Computing weights and averaged value for '+str(day))
        
        # how many stations had readings for this day
        not_null_stations = pd.notnull(station_readings.loc[day,:])==1
        num_stations = len(not_null_stations==True)
        #print(num_stations)
        
        #print(station_readings.loc[day,:])
        
        for station in station_readings.loc[day,:].index:
            if (not_null_stations[station] == True):
                # average distance between this site and others
                total_dist = 0
                for other_station in station_readings.loc[day,:].index:
                    if station != other_station:
                        dist_between_stations = lat_lon_dist([station_list.loc[station]['Latitude'],station_list.loc[station]['Longitude']],[station_list.loc[other_station]['Latitude'],station_list.loc[other_station]['Longitude']])
                        total_dist = total_dist + dist_between_stations
                
                if num_stations is not 1: # so you're not dividing by 0
                    r_jk_bar = total_dist/(num_stations-1)
                else:
                    r_jk_bar = 1
                
                CW_ijk = 1/float(num_stations) + r_jk_bar/station_list.loc[station]['Distance']    
                R_ij = (1/station_list.loc[station]['Distance'] )**2    
                station_weights.loc[day,station] = R_ij * CW_ijk

                # set the weight for the best monitor to 0 if you wanted that
                if ((skip_best==True) and (station==station_list.index[0])):
                    station_weights.loc[day,station] = 0
    
        # perform weighted average of stations for this day 
        weights_sum = 0
        values_sum = 0
        for station in station_readings.loc[day,:].index:
            if pd.notnull(station_readings.loc[day,station]):
                weights_sum = weights_sum + station_weights.loc[day,station]
                values_sum = values_sum + station_readings.loc[day,station]*station_weights.loc[day,station]
            
            #station_weights.loc[day,monitor_info[i]['station_id']] = monitor_info[i]['Weight']
            
        if weights_sum is not 0: # avoid dividing by zero--if no data for any of them, keep it as NaN
            daily_average[day] = values_sum/weights_sum
        else:
            daily_average[day] = np.nan
    
        #print(station_readings)
        #print(station_weights)
        #print(daily_average)
    
    # put data into the air quality object that'll be returned
    aq_obj = aq_data()
    aq_obj.lat_lon = [my_lat,my_lon]
    aq_obj.param = param_code
    aq_obj.r_max = r_max
    aq_obj.monitor_info = daily_monitor_info
    aq_obj.daily_average = daily_average
    aq_obj.station_readings = station_readings
    aq_obj.station_weights = station_weights
    aq_obj.station_list = station_list
        
    return aq_obj
    
def aq_method_plot(aq_obj):
    
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
    #monitor_info = aq_obj.monitor_info
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