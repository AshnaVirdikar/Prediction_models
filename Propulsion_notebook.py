#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras.constraints import maxnorm


# In[3]:


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# In[4]:


#Importing dataset
dataframe = pd.read_csv("propulsion.csv")

dataframe = dataframe.round(3)


# In[6]:


# Assign names to Columns
dataframe.columns = ['lever_position', 'ship_speed', 'gt_shaft', 
                     'gt_rate', 'gg_rate', 'sbp_torque', 'pp_torque',
                     'hpt_temp',  'gt_c_o_temp', 
                     'hpt_pressure',  'gt_c_o_pressure',
                     'gt_exhaust_pressure', 'turbine_inj_control', 
                     'fuel_flow', 'gt_c_decay',  'gt_t_decay']


# In[7]:


#identify missing values
missing_values =dataframe.isnull().sum()
missing_values


# In[8]:


#getting data insights
pd.set_option('display.max_columns', 1000)
print(dataframe.head()) 


# In[9]:


print("Statistical Description:", dataframe.describe())


# In[10]:


print("Shape:", dataframe.shape)
print("Data Types:", dataframe.dtypes)

dataset = dataframe.values


# In[11]:


#splitting dataset 
X = dataset[:,0:15]
Y1 = dataset[:,14] #gt_c_decay
Y2 = dataset[:,15] #gt_t_decay


# In[12]:


#Feature Selection for gt_c_decay
estimator = ExtraTreesRegressor()
rfe = RFE(estimator, 3)
fit = rfe.fit(X, Y1)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 


# In[13]:


#Feature Selection for gt_t_decay
estimator = ExtraTreesRegressor()
rfe = RFE(estimator, 3)
fit = rfe.fit(X, Y2)

print("Number of Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_) 


# In[14]:


#histogram of dataframe

dataframe.hist()


# In[15]:


# Split Data to Train and Test
X_Train, X_Test, Y1_Train, Y1_Test = train_test_split(X, Y1, test_size=0.2)


# In[16]:


num_instances = len(X)

models = []
models.append(('LiR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))


# In[23]:


# Evaluations
results = []
names = []
scoring = []

for name, model in models:
    # Fit the model
    model.fit(X_Train, Y1_Train)
    
    predictions = model.predict(X_Test)
  


# In[25]:


# Evaluate the model
score = explained_variance_score(Y1_Test, predictions)
mae = mean_absolute_error(predictions, Y1_Test)
  
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
results.append(mae)
names.append(name)
msg = "%s: %f (%f)" % (name, score, mae)
print(msg)
   


# In[26]:


# Split Data to Train and Test
X_Train, X_Test, Y2_Train, Y2_Test = train_test_split(X, Y2, test_size=0.2)

num_instances = len(X)


# In[27]:


models = []
models.append(('LiR', LinearRegression()))
models.append(('Ridge', Ridge()))
models.append(('Bag_Re', BaggingRegressor()))
models.append(('RandomForest', RandomForestRegressor()))
models.append(('ExtraTreesRegressor', ExtraTreesRegressor()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))


# In[28]:


# Evaluations
results = []
names = []
scoring = []

for name, model in models:
    # Fit the model
    model.fit(X_Train, Y2_Train)
    
    predictions = model.predict(X_Test)


# In[30]:


# Evaluate the model
score = explained_variance_score(Y2_Test, predictions)
mae = mean_absolute_error(predictions, Y2_Test)
   
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
results.append(mae)
names.append(name)
msg = "%s: %f (%f)" % (name, score, mae)
print(msg)
    


# In[31]:


#Now for  using Deep Learning Models to predict values  
    
# Split Data to Train and Test
X_Train, X_Test, Y1_Train, Y1_Test = train_test_split(X, Y1, test_size=0.3)


# In[32]:


# create model
model = Sequential()
model.add(Dense(6, input_dim=15, init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))


# In[33]:


# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam')


# In[34]:


# Fit the model
model.fit(X_Train, Y1_Train, epochs=100, batch_size=10)


# In[35]:


# Evaluate the model
scores = model.evaluate(X_Test, Y1_Test)
print("score: %.2f%%" % (100-scores))


# In[36]:


# Split Data to Train and Test
X_Train, X_Test, Y2_Train, Y2_Test = train_test_split(X, Y2, test_size=0.3)


# In[37]:


# create model
model = Sequential()
model.add(Dense(6, input_dim=15, init='uniform', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(4, init='uniform', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))


# In[38]:


# Compile model
model.compile(loss='mean_absolute_error', optimizer='adam')


# In[39]:


# Fit the model
model.fit(X_Train, Y2_Train, epochs=100, batch_size=10)


# In[40]:


# Evaluate the model
scores = model.evaluate(X_Test, Y2_Test)
print("score: %.2f%%" % (100-scores))

