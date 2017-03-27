# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:07:18 2017

@author: danjr
"""

import AQ_ML as aq
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np

#station_df = pd.read_csv(aq.station_df_path)
if ~('all_data_c' in locals()):
    all_data_c = pd.read_csv(aq.all_data_path,usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean','Parameter Code','Latitude','Longitude'])
    all_data_c = all_data_c.rename(columns={'Site Num':'Site Number'})

all_data = all_data_c.copy()

latlon = (38.610905,-122.868794)
r_max_interp = 100# how far from latlon of interest should it look for stations?
r_max_ML = 150 # for each station it finds, how far should it look aroud it in imputing the missing values?

start_date = '2011-01-01'
end_date = '2015-06-30'

data,target_data = aq.predict_aq_vals(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data,ignore_closest=True)

from sklearn.metrics import r2_score
r2 = r2_score(data,target_data)