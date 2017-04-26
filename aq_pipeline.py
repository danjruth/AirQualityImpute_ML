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

start_date = '2015-01-01'
end_date = '2015-12-01'

latlon = (36.110693,-80.226438)
r_max_interp = 100 # how far from latlon of interest should it look for stations?
r_max_ML = 200 # for each station it finds, how far should it look aroud it in imputing the missing values?

### ---- END USER INPUTS ---- ###

# get the raw daa

all_data = aq.extract_raw_data(start_date,end_date)

pm25_data = aq.extract_raw_data(start_date,end_date,param_code=88101)
#ozone_data = aq.extract_raw_data(start_date,end_date,param_code=44201)
#CO_data = aq.extract_raw_data(start_date,end_date,param_code=42101)
#other_data = pd.concat([pm25_data,CO_data,ozone_data])
other_data = pd.concat([pm25_data])
other_data = other_data.set_index(pd.Series(data=range(len(other_data))))

# filter out stations that are definitley too far away to be of any use.
# we know nothing farther than r_max_interp+r_max_ML will be used.
# not necessary, but it should save time.
all_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,all_data,start_date,end_date,ignore_closest=False)
other_data = aq.identify_nearby_stations(latlon,r_max_interp+r_max_ML,other_data,start_date,end_date,ignore_closest=False)
all_data = all_data.sort_values('Date Local')
other_data = other_data.sort_values('Date Local')

# run the algorithm
data, target_data, results_noML, station_obj_list, composite_data, orig = aq.predict_aq_vals(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data,other_data,ignore_closest=True,return_lots=True)

# construct dataframe to facilitate comparison between methods
compare_df = pd.DataFrame()
compare_df['predicted'] = data
compare_df['predicted_noML'] = results_noML
compare_df['target'] = target_data
compare_df_all = compare_df.copy()
# only keep rows for which there is target data to compare against; fill missing predicted values with 0?
compare_df = compare_df[np.isfinite(compare_df['target'])]
compare_df = compare_df.fillna(0) # think more about this

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

# plot the results against target data
fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(211)
ax1.plot(results_noML,'.-',color='gray',label='predicted, no ML')
ax1.plot(data,'.-',color='k',label='predicted')
ax1.plot(target_data,'.-',color='green',label='target')
ax1.legend()
ax1.set_title('Instantaneous')
plt.show()

# plot the results against target data
ax2 = fig.add_subplot(212,sharex=ax1,sharey=ax1)
ax2.plot(results_noML.rolling(window=win,min_periods=0).mean(),'.-',color='gray',label='predicted, no ML')
ax2.plot(data.rolling(window=win,min_periods=0).mean(),'.-',color='k',label='predicted')
ax2.plot(target_data.rolling(window=win,min_periods=0).mean(),'.-',color='green',label='target')
ax2.set_title('Rolling')
ax2.legend()
ax1.set_ylabel('PM10 Concentration')
ax2.set_ylabel('PM10 Concentration')
plt.show()

# Print error
plt.figure()
plt.plot(data-target_data,label='error')
plt.plot(results_noML-target_data,label='error, no ML')
plt.legend()
plt.show()

# one against the other
plt.figure()
plt.scatter(compare_df['target'],compare_df['predicted'],label='with ML')
plt.scatter(compare_df['target'],compare_df['predicted_noML'],label='no ML')
plt.plot([0,0],[data.max(),data.max()])
plt.legend()
plt.ylabel('Predicted')
plt.xlabel('Target')
plt.show()



# make a nice plot
