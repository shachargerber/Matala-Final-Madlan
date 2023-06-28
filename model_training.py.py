#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sklearn as sk
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse
import pickle
import openpyxl
from madlan_data_prep import prepare_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet


path='output_all_students_Train_v10.xlsx'

def model_elastic(path):
    
    df_model,x,y=prepare_data(path)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
    elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.5)
    elastic_net.fit(x_train, y_train)
    
    elastic_net_cross =ElasticNetCV(cv=10)
    mse_scores = -cross_val_score(elastic_net_cross, x, y, cv=10, scoring='neg_mean_squared_error')
    rmse_scores=np.sqrt(mse_scores)
    rmse_mean = np.mean(rmse_scores)
    
    mae_scores = -cross_val_score(elastic_net_cross, x, y, cv=10, scoring='neg_mean_absolute_error')
    mae_mean=np.mean(mae_scores)
    print("The RMSE mean is of the cross validation is: ",rmse_mean)
    print("The MAE mean is of the cross validation is: ",mae_mean)
    
    prediction = elastic_net.predict(x_test)
    
      # Save the trained model as PKL
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(elastic_net, f)

    return prediction

print(model_elastic(path))


