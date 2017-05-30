# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:01:38 2017

@author: druth
"""

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

r_max = r_max_interp+r_max_ML

fig = plt.figure(figsize=(15, 8), facecolor='w')    

# set viewing window for map plot
scale_factor = 100.0 # lat/lon coords to show from center point per km of r_max
left_lim = latlon[1]-1.2*r_max/scale_factor
right_lim = latlon[1]+1.2*r_max/scale_factor
bottom_lim = latlon[0]-r_max/scale_factor
top_lim = latlon[0]+r_max/scale_factor

m = Basemap(projection='merc',resolution='i',lat_0=latlon[0],lon_0=latlon[1],llcrnrlon=left_lim,llcrnrlat=bottom_lim,urcrnrlon=right_lim,urcrnrlat=top_lim)
m.shadedrelief()
m.drawstates()
m.drawcountries()
m.drawrivers()
m.drawcoastlines()

plt.show()

param_colors = {88101:'g',44201:'c',42101:'m'}

# plot each EPA site on the map, and connect it to the soiling station with a line whose width is proportional to the weight

for station in station_obj_list:

    for nearby_station_indx in station.nearby_stations.index:
        
        if nearby_station_indx in station.gs.columns:
            
            nearby_station = station.nearby_stations.loc[nearby_station_indx]
            (x,y) = m([station.latlon[1],nearby_station['Longitude']],[station.latlon[0],nearby_station['Latitude']])
            m.plot(x,y,'-',color='k',lw=1,alpha=0.5)
            
            (x,y) = m(nearby_station['Longitude'],nearby_station['Latitude'])
            m.plot(x,y,'^',color='k',ms=5,alpha=0.5)
        
    for other_station_indx in station.other_stations.index:
        
        if other_station_indx in station.gs.columns:
            
            other_station = station.other_stations.loc[other_station_indx]
            (x,y) = m([station.latlon[1],other_station['Longitude']],[station.latlon[0],other_station['Latitude']])
            m.plot(x,y,'-',color = param_colors[other_station['Parameter Code']],lw=1,alpha=0.5)
            
            (x,y) = m(other_station['Longitude'],other_station['Latitude'])
            m.plot(x,y,'^',color = param_colors[other_station['Parameter Code']],ms=5,alpha=0.5)
        
    for station in station_obj_list:
        (x,y) = m(station.latlon[1],station.latlon[0])
        m.plot(x,y,'o',color = 'k',ms=9)
        
        (x,y) = m([station.latlon[1],latlon[1]],[station.latlon[0],latlon[0]])
        m.plot(x,y,'-',color='k',lw=2)
            
        (x,y) = m(latlon[1],latlon[0])
        m.plot(x,y,'x',color = 'r',lw=5,ms=20)
