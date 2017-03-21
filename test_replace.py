# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:06:02 2017

@author: danjr
"""


import replace_nans_with_ML as ml
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import pickle
file = open("losangeles_test.obj",'rb')
aq_obj = pickle.load(file)
file.close()

col_to_replace=2

#original_vals = pd.Series()

#original_vals = np.array()
original_vals = np.array(aq_obj.station_readings.iloc[:,col_to_replace])
original_series = (aq_obj.station_readings.iloc[:,col_to_replace])
original_series_locked = original_series
#print(original_series)
#stophere
#to_plot_original = 

#print('Original df:')
#print(aq_obj.station_readings)

obj_for_ml = aq_obj

model = ml.create_ML_model(obj_for_ml.station_readings,col_to_replace)
station_readings_corrected = ml.replace(obj_for_ml.station_readings,col_to_replace,model)
corrected_only = station_readings_corrected.ix[np.isnan(original_vals),col_to_replace]
not_corrected = station_readings_corrected.ix[np.isfinite(original_vals),col_to_replace]

plt.figure()
plt.plot(original_series,label='final data',lw=1,color='k')
plt.plot(corrected_only,'.',markersize=10,label='added with ML',color='r')
plt.plot(not_corrected,'.',markersize=10,label='original',color='g')

plt.legend()
plt.show()



#print('Original:')
#print(aq_obj.station_readings.ix[:,col_to_replace])
#print('Corrected:')
#print(corrected_only)