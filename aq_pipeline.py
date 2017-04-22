# -*- coding: utf-8 -*-
"""

Runs the algorithm implemented in AQ_ML.py

Created on Sun Mar 12 19:07:18 2017

@author: danjr
"""

import AQ_ML as aq
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

### ---- USER INPUTS ---- ###

start_date = '2013--01-01'
end_date = '2016-01-01'

latlon = (	45.028620,	-92.783360)
r_max_interp = 100 # how far from latlon of interest should it look for stations?
r_max_ML = 200 # for each station it find○s, how far should it look aroud it in imputing the missing values?

### ---- END USER INPUTS ---- ###

# get the raw daa
all_data = aq.extract_raw_data(start_date,end_date)

# run the algorithm
data, target_data, results_noML, station_obj_list, composite_data, orig = aq.predict_aq_vals(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data,ignore_closest=True,return_lots=True)

# plot the results against target data
plt.figure()
plt.plot(data,'.-',label='predicted')
plt.plot(results_noML,'.-',label='predicted, no ML')
plt.plot(target_data,'.-',label='target')
plt.legend()
plt.show()

# construct dataframe to facilitate comparison between methods
compare_df = pd.DataFrame()
compare_df['predicted'] = data
compare_df['predicted_noML'] = results_noML
compare_df['target'] = target_data
compare_df_all = compare_df.copy()
compare_df = compare_df[np.isfinite(compare_df['target'])]
compare_df = compare_df.fillna(0)

## Compute/print some metrics

from sklearn.metrics import r2_score, mean_absolute_error

# window for the rolling mean calculations.
# rolling mean is useful since values reported for soiling periods are averages
# over at least 14 days.
win = 14

print('----- METRICS -----')

r2 = r2_score(compare_df['predicted'],compare_df['target'])
r2_noML = r2_score(compare_df['predicted_noML'],compare_df['target'])
print('R squareds (with, without ML) are:')
print(r2,r2_noML)

r2_roll = r2_score(compare_df['predicted'].rolling(window=win,min_periods=0).mean(),compare_df['target'].rolling(window=win,min_periods=0).mean())
r2_roll_noML = r2_score(compare_df['predicted_noML'].rolling(window=win,min_periods=0).mean(),compare_df['target'].rolling(window=win,min_periods=0).mean())
print('Rolling r squareds (with, without ML) are:')
print(r2_roll,r2_roll_noML)

corr = compare_df['target'].corr(compare_df['predicted'])
corr_noML = compare_df['target'].corr(compare_df['predicted_noML'])
print('Pearson correlations (with, without ML) are:')
print(corr,corr_noML)

corr_roll = compare_df['target'].rolling(window=win,min_periods=0).mean().corr(compare_df['predicted'].rolling(window=win,min_periods=0).mean())
corr_roll_noML = compare_df['target'].rolling(window=win,min_periods=0).mean().corr(compare_df['predicted_noML'].rolling(window=win,min_periods=0).mean())
print('Rolling Pearson correlations (with, without ML) are:')
print(corr_roll,corr_roll_noML)

mae = mean_absolute_error(compare_df['predicted'],compare_df['target'])
mae_noML = mean_absolute_error(compare_df['predicted_noML'],compare_df['target'])
print('Mean abs. errors (with, without ML) are:')
print(mae,mae_noML)

mae_roll = mean_absolute_error(compare_df['predicted'].rolling(window=win,min_periods=0).mean(),compare_df['target'].rolling(window=win,min_periods=0).mean())
mae_roll_noML = mean_absolute_error(compare_df['predicted_noML'].rolling(window=win,min_periods=0).mean(),compare_df['target'].rolling(window=win,min_periods=0).mean())
print('Rolling mean abs. errors (with, without ML) are:')
print(mae_roll,mae_roll_noML)

# Print error
plt.figure()
plt.plot(data-target_data,label='error')
plt.plot(results_noML-target_data,label='error, no ML')
plt.legend()
plt.show()