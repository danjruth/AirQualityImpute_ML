import sklearn.tree
import sklearn.neural_network
import sklearn.linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import read_airQuality as aq

import pickle
file = open("losangeles_test.obj",'rb')
aq_obj = pickle.load(file)
file.close()

#aq.aq_method_plot(aq_obj)

data_shape = np.shape(aq_obj.station_readings)
num_cols = data_shape[1]
num_cols = 30

# which column will be predicted?
out_col = 2

# which columns are the predictors?
all_cols = np.array(range(0,num_cols))
in_cols = all_cols[(all_cols!=out_col)]
in_cols = [0,1,3,7,8,12,14,16]

# which rows have values for the out column
have_out_vals = np.isnan(aq_obj.station_readings.iloc[:,out_col])
have_out_vals = np.where(have_out_vals==False)[0]
num_known = len(have_out_vals)
print('Predicting column '+str(out_col)+' with '+str(num_known)+' known rows.')

train_indx = range(0,int(num_known*.75))
test_indx = range(int(num_known*.75),num_known)

# separate inputs and outputs
inputs = aq_obj.station_readings.iloc[have_out_vals,in_cols].values
output = aq_obj.station_readings.iloc[have_out_vals,out_col].values

inputs_before_transform = inputs

# replace missing
import sklearn.preprocessing
imp = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
inputs = sklearn.preprocessing.normalize(imp.fit_transform(inputs))
#output = sklearn.preprocessing.normalize(imp.fit_transform(output)).transpose()

# normalize data
'''
import scipy.stats
plt.figure()
plt.subplot(2,1,1)
plt.plot(inputs[:,0])
plt.plot(output,'.')
#inputs = scipy.stats.zscore(inputs,axis=0)
#outupt = scipy.stats.zscore(output)
plt.subplot(2,1,2)
plt.plot(inputs[:,0])
plt.plot(output,'.')
plt.show()
'''


# shuffle rows
from sklearn.utils import shuffle
inputs_noshuffle = inputs
inputs,output = shuffle(inputs,output)
output = output.ravel()

# NN params
hl_size = (5,3)
act = 'tanh'

'''
from sklearn.preprocessing import StandardScaler
inputs = StandardScaler().fit_transform(inputs)
output = StandardScaler().fit_transform(output)
'''

# create models
nn_model = sklearn.neural_network.MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=hl_size,activation=act)
linear_model = sklearn.linear_model.LinearRegression()
rtree = sklearn.tree.DecisionTreeRegressor(min_samples_leaf=10,max_features=0.50)

# fit models to data
nn_model.fit(inputs[train_indx,:], output[train_indx])
linear_model.fit(inputs[train_indx,:], output[train_indx])
rtree.fit(inputs[train_indx,:], output[train_indx])

# predict with model
nn_predicted = nn_model.predict(inputs[test_indx])
linear_predicted = linear_model.predict(inputs[test_indx])
rtree_predicted = rtree.predict(inputs[test_indx])

nn_known_predicted = nn_model.predict(inputs[train_indx])
linear_known_predicted = linear_model.predict(inputs[train_indx])
rtree_known_predicted = rtree.predict(inputs[train_indx])

# target vs predicted
plt.figure()
plt.plot(output[test_indx],nn_predicted,'.',label='Neural net',color='r')
plt.plot(output[test_indx],linear_predicted,'.',label='Linear model',color='b')
plt.plot(output[test_indx],rtree_predicted,'.',label='Regression tree',color='g')
plt.plot(output[train_indx],nn_known_predicted,'x',color='r')
plt.plot(output[train_indx],linear_known_predicted,'x',color='b')
plt.plot(output[train_indx],rtree_known_predicted,'x',color='g')
plt.plot([0, np.max(output)],[0, np.max(output)],color='k')
plt.xlabel('Target')
plt.ylabel('Predicted')
plt.legend(loc=4)
plt.show()

# r2 score
from sklearn.metrics import r2_score
print('R-squared value for the neural network is test: '+str(r2_score(output[test_indx],nn_predicted))+', train: '+str(r2_score(output[train_indx],nn_known_predicted)))
print('R-squared value for the linear model is test: '+str(r2_score(output[test_indx],linear_predicted))+', train: '+str(r2_score(output[train_indx],linear_known_predicted)))
print('R-squared value for the regression tree is test: '+str(r2_score(output[test_indx],rtree_predicted))+', train: '+str(r2_score(output[train_indx],rtree_known_predicted)))


plt.matshow(nn_model.coefs_[0])
plt.colorbar()
plt.show()

plt.matshow(nn_model.coefs_[1])
plt.colorbar()
plt.show()

plt.figure()
if len(hl_size) == 1:
    imp = list()
    for i in range(0,num_cols-1):
        imp.append(np.sum((np.multiply(nn_model.coefs_[0][i,:],nn_model.coefs_[1]))))
    

    
    plt.subplot(3,1,1)
    plt.bar(range(0,num_cols-1),imp)
    plt.ylabel('Neural network')
    
plt.title('Predictor importances')

    
plt.subplot(3,1,2)
imp_linear = linear_model.coef_
plt.bar(range(0,len(in_cols)),imp_linear.transpose())
plt.ylabel('Linear model')

plt.subplot(3,1,3)
imp_rtree = rtree.feature_importances_
plt.bar(range(0,len(in_cols)),imp_rtree)
plt.ylabel('Regression tree')

plt.show()
