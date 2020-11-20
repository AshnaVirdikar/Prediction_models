# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:10:16 2020

@author: Ashna
"""

#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE # feature elimination
from sklearn.ensemble import ExtraTreesRegressor #importing estimator


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
tf.__version__

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.constraints import maxnorm


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#Importing dataset
dataframe = pd.read_csv("propulsion.csv")

dataframe = dataframe.round(3)

# Assign names to Columns
dataframe.columns = ['lever_position', 'ship_speed', 'gt_shaft', 
                     'gt_rate', 'gg_rate', 'sbp_torque', 'pp_torque',
                     'hpt_temp', 'gt_c_i_temp', 'gt_c_o_temp', 
                     'hpt_pressure', 'gt_c_i_pressure', 'gt_c_o_pressure',
                     'gt_exhaust_pressure', 'turbine_inj_control', 
                     'fuel_flow', 'gt_c_decay',  'gt_t_decay']

#identify missing values
missing_values =dataframe.isnull().sum()
missing_values

#drop data 

dataframe = dataframe.drop(axis=1,columns='gt_c_i_temp')
dataframe = dataframe.drop(axis=1,columns='gt_c_i_pressure')

#getting data insights
pd.set_option('display.max_columns', 1000)
print(dataframe.head()) 

print("Statistical Description:", dataframe.describe())


print("Shape:", dataframe.shape)
print("Data Types:", dataframe.dtypes)

dataset = dataframe.values


#splitting dataset 
X = dataset[:,0:15]
Y1 = dataset[:,14] #gt_c_decay
Y2 = dataset[:,15] #gt_t_decay

#Feature Selection for gt_c_decay
estimator = ExtraTreesRegressor()
rfe = RFE(estimator, 3)
fit = rfe.fit(X, Y1)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 


#'gt_rate'(Gas Turbine rate of revolutions),
#'gt_c_o_temp'(GT Compressor outlet air temperature ) and 
#'fuel_flow' were top 3 selected features/feature combination for predicting 
# 'gt_c_decay'(GT Compressor decay state coefficient)

 #Feature Selection for gt_t_decay
estimator = ExtraTreesRegressor()
rfe = RFE(estimator, 3)
fit = rfe.fit(X, Y2)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 

#'gg_rate', 'gt_c_o_pressure' and 'turbine_inj_control'(Turbine Injecton Control (TIC) [%]) 
#were top 3 selected features/feature combination for predicting 
#'gt_t_decay'(GT Turbine decay state coefficient)

#histogram of dataframe

dataframe.hist()


# Split Data to Train and Test
X_Train, X_Test, Y1_Train, Y1_Test = train_test_split(X, Y1, test_size=0.2)

num_instances = len(X)

models = []
models.append(('LiR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))

# Evaluations
results = []
names = []
scoring = []

for name, model in models:
    # Fit the model
    model.fit(X_Train, Y1_Train)
    
    predictions = model.predict(X_Test)
    
    # Evaluate the model
    score = explained_variance_score(Y1_Test, predictions)
    mae = mean_absolute_error(predictions, Y1_Test)
   
    # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    results.append(mae)
    names.append(name)
    msg = "%s: %f (%f)" % (name, score, mae)
    print(msg)
    
    #Extra Trees Regressor' and 'Random Forest Regressor' are the best 
    #estimators/models  for predicting 'gt_c_decay'.
    
    # Split Data to Train and Test
X_Train, X_Test, Y2_Train, Y2_Test = train_test_split(X, Y2, test_size=0.2)

num_instances = len(X)

models = []
models.append(('LiR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))

# Evaluations
results = []
names = []
scoring = []

for name, model in models:
    # Fit the model
    model.fit(X_Train, Y2_Train)
    
    predictions = model.predict(X_Test)
    
    # Evaluate the model
    score = explained_variance_score(Y2_Test, predictions)
    mae = mean_absolute_error(predictions, Y2_Test)
   
 # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    
results.append(mae)
names.append(name)
msg = "%s: %f (%f)" % (name, score, mae)
print(msg)
    
    #'Extra Trees Regressor' and 'Bagging Regressor' are the best 
    #estimators/models for for predicting 'gt_t_decay'.
    
#Now for  using Deep Learning Models to predict values  
    
# Split Data to Train and Test
X_Train, X_Test, Y1_Train, Y1_Test = train_test_split(X, Y1, test_size=0.3)




# create model
model = Sequential()
model.add(Dense(6, input_dim=15, init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))

# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam')

# Fit the model
model.fit(X_Train, Y1_Train, epochs=100, batch_size=10)

# Evaluate the model
scores = model.evaluate(X_Test, Y1_Test)
print("score: %.2f%%" % (100-scores))

# Split Data to Train and Test
X_Train, X_Test, Y2_Train, Y2_Test = train_test_split(X, Y2, test_size=0.3)




# create model
model = Sequential()
model.add(Dense(6, input_dim=15, init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))

# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam')

# Fit the model
model.fit(X_Train, Y2_Train, epochs=100, batch_size=10)

# Evaluate the model
scores = model.evaluate(X_Test, Y2_Test)
print("score: %.2f%%" % (100-scores))









