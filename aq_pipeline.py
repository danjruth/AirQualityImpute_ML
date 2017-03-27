# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:07:18 2017

@author: danjr
"""

import AQ_ML as aq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#station_df = pd.read_csv(aq.station_df_path)
if ~('all_data_c' in locals()):
    all_data_c = pd.read_csv(aq.all_data_path,usecols=['State Code','County Code','Site Num','Date Local','Arithmetic Mean','Parameter Code','Latitude','Longitude'])
    all_data_c = all_data_c.rename(columns={'Site Num':'Site Number'})

all_data = all_data_c.copy()

latlon = (32.414344,-111.154544)
r_max_interp = 50 # how far from latlon of interest should it look for stations?
r_max_ML = 75 # for each station it finds, how far should it look aroud it in imputing the missing values?

start_date = '2012-01-01'
end_date = '2015-06-30'

data, target_data, results_noML = aq.predict_aq_vals(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data,ignore_closest=True)

plt.figure()
plt.plot(data,'.-',label='predicted')
plt.plot(results_noML,'.-',label='predicted, no ML')
plt.plot(target_data,'.-',label='target')
plt.legend()
plt.show()

compare_df = pd.DataFrame()
compare_df['predicted'] = data
compare_df['predicted_noML'] = results_noML
compare_df['target'] = target_data
compare_df = compare_df[np.isfinite(compare_df['target'])]

from sklearn.metrics import r2_score
r2 = r2_score(compare_df['predicted'],compare_df['target'])
r2_noML = r2_score(compare_df['predicted_noML'],compare_df['target'])
print('R squareds (with, without ML) are:')
print(r2,r2_noML)