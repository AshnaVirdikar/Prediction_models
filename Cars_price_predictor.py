# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:18:49 2020

@author: Ashna
"""

#Importing libraries 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
import os


#Importing the dataset and getting first intuitions
cars = pd.read_csv("cars_predictor.csv")
pd.set_option('display.max_columns', 1000)
cars.head()
 
#Identifiying missing values
missing_values = cars.isnull().sum()
missing_values


# check of numerical attributes
cars.describe()

#checking for values of categorical attributes 
cat_val = ["make", "model", "condition", "fuel_type", "color",
          "drive_unit","segment"]

for col in cat_val:
    print ([col]," : ",cars[col].unique())

#Visualization to understand how to clean data
# Filter bad data
cars_c = cars.copy()

#scattermatrix plot
num_attributes = ["priceUSD", "year", "mileage(kilometers)", "volume(cm3)"]
%matplotlib inline
pd.plotting.scatter_matrix(cars_c[num_attributes], figsize = (12,8), alpha = 0.1)


#Histogram of priceUSD
cars_c["priceUSD"].hist(bins = 20, log = True)


#Data Cleansing
# Fresh copy
cars_clean = cars.copy()

# Filter bad data
cars_clean = cars_clean[
    (cars_clean["year"].between(1980, 2019, inclusive=True)) &
    (cars_clean["mileage(kilometers)"].between(0, 10, inclusive=True)) &
    (cars_clean["priceUSD"].between(2379, 195000, inclusive=True)) 
    
    ]


# Replace the NaN-Values
cars_clean['volume(cm3)'].fillna(value=2103.96, inplace=True)
cars_clean['drive_unit'].fillna(value='front-wheel drive', inplace=True)
cars_clean['segment'].fillna(value='J', inplace=True)


# Change categorical attributes dtype to category

for col in cars_clean:
    if cars_clean[col].dtype == "object":
        cars_clean[col] = cars_clean[col].astype('category')

# Assign codes to categorical attribues instead of strings
cat_columns = cars_clean.select_dtypes(['category']).columns

cars_clean[cat_columns] = cars_clean[cat_columns].apply(lambda x: x.cat.codes)
    
# Drop probably unuseful columns

drop_cols = ["color"]
cars_clean = cars_clean.drop(drop_cols, axis=1)
cars_clean.head()



# Getting the train and test sets
train_set, test_set = train_test_split(cars_clean, test_size = 0.2, random_state = 42)

# Seperation of Predictors (Features) and the Labels (Targets)

cars_price = train_set["priceUSD"].copy()
cars = train_set.drop("priceUSD", axis=1)


#Custom-Transformers and Pipelines
# Create a class to select numerical or categorical columns 

class DFSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Setting categorical and numerical attributes

cat_attribs = ["make", "model", "condition", "fuel_type", "drive_unit", "segment"]
num_attribs = list(cars.drop(cat_attribs, axis=1))

# Building the Pipelines

num_pipeline = Pipeline([
    ("selector", DFSelector(num_attribs)),
    ("std_scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("selector", DFSelector(cat_attribs)),
    ("encoder", OneHotEncoder(sparse=True))
])

full_pipeline = FeatureUnion(transformer_list =[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

cars_prepared = full_pipeline.fit_transform(cars_clean[:128])

# Training and Comparing Models
#LINEAR REGRESSION MODEL
lin_reg = LinearRegression()
lin_reg.fit(cars_prepared, cars_price)

from sklearn.metrics import mean_squared_error
cars_predictions = lin_reg.predict(cars_prepared)
lin_mse = mean_squared_error(cars_price, cars_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

cars_predictions[0:4]

list(cars_price[0:4])

#DECISION TREE REGRESSION MODEL
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(cars_prepared, cars_price)

cars_predictions = tree_reg.predict(cars_prepared)
tree_mse = mean_squared_error(cars_price, cars_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

#RANDOM FOREST REGRESSION MODEL
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42, n_jobs =-1, max_depth = 30 )
forest_reg.fit(cars_prepared, cars_price)
cars_predictions = forest_reg.predict(cars_prepared)

forest_mse = mean_squared_error(cars_price, cars_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

#LinearRegression does very well. DecisionTree is either perfect or it overfitted badly.
#CrossValidation will show us the truth. RandomFroest is slightly better than 
#LinearRegression. I played a little bith with the "max_depth" to have a good 
#comprise between result and computation power.

#Cross-Validation

from sklearn.model_selection import cross_val_score

def display_scores(scores):
    
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

#LinReg - CrossValidation
# Offline i used CV=10

scores = cross_val_score(lin_reg, cars_prepared, cars_price,
                         scoring="neg_mean_squared_error", cv=4)
lin_rmse_scores = np.sqrt(-scores)

display_scores(lin_rmse_scores)

#DecissionTree - CrossValidation
# Offline i used CV=10

scores = cross_val_score(tree_reg, cars_prepared, cars_price,
                         scoring="neg_mean_squared_error", cv=4)
tree_rmse_scores = np.sqrt(-scores)

display_scores(tree_rmse_scores)


#RandomForest - CrossValidation
# Offline i used CV=8

from sklearn.model_selection import cross_val_score

scores = cross_val_score(forest_reg, cars_prepared, cars_price,
                         scoring="neg_mean_squared_error", cv=2)
forest_rmse_scores = np.sqrt(-scores)

display_scores(forest_rmse_scores)


#Cross-Validation reveals, that LinearRegression is doing way worse,
#especially the DecisionTree. RandomForest seems to be the best alternative.

# Feature Importance

feature_importances = forest_reg.feature_importances_
feature_importances
cat_encoder = cat_pipeline.named_steps["encoder"]

#cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs #+ cat_encoder
sorted(zip(feature_importances, attributes), reverse=True)

#Final Prediction and conclusion

final_model = forest_reg

cars_test = test_set.drop("priceUSD", axis = 1)
cars_price_test = test_set["priceUSD"].copy()


#Evaluation how good the model fit is
from sklearn.metrics import mean_squared_error

final_error=  mean_squared_error(cars_price_test, cars_predictions[:33])

final_rmse = np.sqrt(final_error)

