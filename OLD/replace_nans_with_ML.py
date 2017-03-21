#import sklearn.tree
import sklearn.neural_network
#import sklearn.linear_model
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import read_airQuality as aq

# create a ML model for a station whose data is already in a df
def create_ML_model(station_readings,col_to_replace,num_cols=None):
    
    data_shape = np.shape(station_readings)
    num_cols = data_shape[1]
    
    # which column will be predicted?
    out_col = col_to_replace
    
    # which columns are the predictors?
    all_cols = np.array(range(0,num_cols))
    #in_cols = all_cols[(all_cols!=out_col)]
    in_cols = [0,1,3,7,8,12,14,16]
    
    # which rows have values for the out column
    have_out_vals = np.isnan(station_readings.iloc[:,out_col])
    have_out_vals = np.where(have_out_vals==False)[0]
    num_known = len(have_out_vals)
    print('Predicting column '+str(out_col)+' with '+str(num_known)+' known rows.')
    
    train_indx = range(0,int(num_known*.75))
    test_indx = range(int(num_known*.75),num_known)
    
    # separate inputs and outputs
    inputs = station_readings.iloc[have_out_vals,in_cols].values
    output = station_readings.iloc[have_out_vals,out_col].values
        
    # replace missing
    import sklearn.preprocessing
    imp = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    inputs = sklearn.preprocessing.normalize(imp.fit_transform(inputs))    
    
    # shuffle rows
    from sklearn.utils import shuffle
    #inputs_noshuffle = inputs
    inputs,output = shuffle(inputs,output)
    output = output.ravel()
    
    # NN params
    hl_size = (5,3)
    act = 'relu'

    
    # create/fit model
    nn_model = sklearn.neural_network.MLPRegressor(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=hl_size,activation=act)
    nn_model.fit(inputs[train_indx,:], output[train_indx])
    
    nn_predicted = nn_model.predict(inputs[test_indx,:])
    nn_known_predicted = nn_model.predict(inputs[train_indx,:])
    
    # output vs target
    plt.figure()
    
    plt.plot(output[test_indx],nn_predicted,'.',label='test',color='k')
    plt.plot(output[train_indx],nn_known_predicted,'x',label='train',color='b')
    plt.plot([0, np.max(output)],[0, np.max(output)],color='k')
    plt.xlabel('Target')
    plt.ylabel('Predicted')
    plt.legend(loc=4)    
    
    plt.show()
    
    # r^2 score
    from sklearn.metrics import r2_score
    print('R-squared value for the model is test: '+str(r2_score(output[test_indx],nn_predicted))+', train: '+str(r2_score(output[train_indx],nn_known_predicted)))

    return nn_model
    
    
# once a model 
def replace(station_readings,col_to_replace,model):
    
    out_col = col_to_replace
    
    data_shape = np.shape(station_readings)
    num_cols = data_shape[1]
    
    
    # determine indxs of NAN values to predict
    need_out_vals = np.isnan(station_readings.iloc[:,out_col])
    need_out_vals = np.where(need_out_vals==True)[0]
    print(need_out_vals)
    print(len(need_out_vals))

    # which columns are the predictors?
    all_cols = np.array(range(0,num_cols))
    #in_cols = all_cols[(all_cols!=out_col)]
    in_cols = [0,1,3,7,8,12,14,16]

    # get inputs for the model
    inputs = station_readings.iloc[need_out_vals,in_cols].values

    # replace missing
    import sklearn.preprocessing
    imp = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    inputs = sklearn.preprocessing.normalize(imp.fit_transform(inputs))    
    
    # predict values
    output = model.predict(inputs)
    #print(output)
    #print(len(output))
    
    # replace NaNs in the original data with the model's predictions
    #print(station_readings.iloc[:,out_col])
    station_readings_out = station_readings
    station_readings_out.iloc[need_out_vals,out_col] = output
    
    #print(station_readings.iloc[:,out_col])

    return station_readings_out
    
# test it out!


