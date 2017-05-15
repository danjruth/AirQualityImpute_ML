# -*- coding: utf-8 -*-
"""
Created on Wed May 10 08:38:22 2017

@author: druth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
import pickle
file = open('C:\Users\druth\Documents\\validation_results.obj','rb')
object_file = pickle.load(file)
file.close()


results = object_file[0].drop('Comments',1).dropna(axis=0,how='any')
results = results.set_index(np.arange(len(results)))
'''

results = pd.read_csv('validation_170515_11-15.csv')

#results = results.drop(results.columns[2],1).dropna(axis=0,how='any')
results = results.set_index(np.arange(len(results)))


fig = plt.figure(figsize=(9,8))
ax_r2 = fig.add_subplot(321)
ax_r2_roll = fig.add_subplot(322)
ax_corr = fig.add_subplot(323)
ax_corr_roll = fig.add_subplot(324)
ax_mae = fig.add_subplot(325)
ax_mae_roll = fig.add_subplot(326)

c = 'k'
c_no = 'gray'
w =0.33

for ix in results.index:
    ax_r2.bar(ix-w,results.loc[ix,'r2'],w,color=c)
    ax_r2.bar(ix,results.loc[ix,'r2_noML'],w,color=c_no)
    
    ax_r2_roll.bar(ix-w,results.loc[ix,'r2_roll'],w,color=c)
    ax_r2_roll.bar(ix,results.loc[ix,'r2_roll_noML'],w,color=c_no)
    
    ax_corr.bar(ix-w,results.loc[ix,'corr'],w,color=c)
    ax_corr.bar(ix,results.loc[ix,'corr_noML'],w,color=c_no)
    
    ax_corr_roll.bar(ix-w,results.loc[ix,'corr_roll'],w,color=c)
    ax_corr_roll.bar(ix,results.loc[ix,'corr_roll_noML'],w,color=c_no)
    
    ax_mae.bar(ix-w,results.loc[ix,'mae'],w,color=c)
    ax_mae.bar(ix,results.loc[ix,'mae_noML'],w,color=c_no)
    
    ax_mae_roll.bar(ix-w,results.loc[ix,'mae_roll'],w,color=c)
    ax_mae_roll.bar(ix,results.loc[ix,'mae_roll_noML'],w,color=c_no)
    
ax_r2.set_ylim([-0.2,1])
ax_r2_roll.set_ylim([-0.2,1])
#ax_corr.set_ylim([-0.2,1])
#ax_corr_roll.set_ylim([-0.2,1])

[ax.xaxis.set_ticklabels([]) for ax in [ax_r2,ax_r2_roll,ax_corr,ax_corr_roll]]
[ax.yaxis.set_ticklabels([]) for ax in [ax_r2_roll,ax_corr_roll,ax_mae_roll]]

ax_r2.set_title('Instantaneous')
ax_r2_roll.set_title('Rolling')
ax_r2.set_ylabel('r2')
ax_corr.set_ylabel('Pearson')
ax_mae.set_ylabel('MAE')

[ax.axhline(0,color='k') for ax in [ax_r2,ax_r2_roll]]

ax_mae.set_xlabel('Test location')
ax_mae.tick_params(axis='x',which='minor',bottom='on')
ax_mae_roll.set_xlabel('Test location')

#fig.savefig('C:\Users\druth\Documents\SULI\Paper\Figures\\aq_validation_bar.pdf')





from mpl_toolkits.basemap import Basemap
fig_map = plt.figure(figsize=(9,6))

m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
        projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

m.shadedrelief()
m.drawstates()
m.drawcountries()
#m.drawrivers()
m.drawcoastlines()

plt.show()

labellist = [None]*len(results)

for ix in results.index:
    (x,y) = m(results.loc[ix,'Longitude'],results.loc[ix,'Latitude'])
    plt.plot(x,y,'x',color='r')
    labellist[ix]=plt.text(x,y,'  '+str(ix)+'  ',horizontalalignment='left',size='large',color='k')
    
'''
labellist[8].set_horizontalalignment('right')
labellist[9].set_horizontalalignment('right')
labellist[9].set_verticalalignment('bottom')
#labellist[11].set_verticalalignment('bottom')
labellist[11].set_text('')
labellist[1].set_text('  1, 11  ')
'''
plt.show()
#fig_map.savefig('C:\Users\druth\Documents\SULI\Paper\Figures\\aq_validation_map.png')