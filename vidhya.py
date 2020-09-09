# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:20:38 2020

@author: LENOVO
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#DATA IMPORTING

df = pd.read_csv('F:/pyWork/pyData/train_v9rqX0R.csv')
df.shape
#DATA CLEANING

df.drop(["Outlet_Identifier","Item_Identifier","Item_Type"],axis=1,inplace=True)
mean=df["Item_Weight"].mean()
df["Item_Weight"].replace(np.nan,mean,inplace=True)
df["Outlet_Establishment_Year"]=2020-df["Outlet_Establishment_Year"]

df["Outlet_Size"].replace(np.nan,"Medium",inplace=True)
df["Item_Fat_Content"].replace("LF","Low Fat",inplace=True)
df["Item_Fat_Content"].replace("low fat","Low Fat",inplace=True)
df["Item_Fat_Content"].replace("reg","Regular",inplace=True)

#DATA WRANGLING


size_dummy = pd.get_dummies(df["Outlet_Size"],drop_first=True)
size_dummy.head(5)

location_dummy = pd.get_dummies(df["Outlet_Location_Type"],drop_first=True)
location_dummy.head(5)

type_dummy = pd.get_dummies(df["Outlet_Type"],drop_first=True)
type_dummy.head(5)

content_dummy = pd.get_dummies(df["Item_Fat_Content"],drop_first=True)
content_dummy.head(5)

df = pd.concat([df,size_dummy,location_dummy,type_dummy,content_dummy],axis=1)
df.head(5)

df.drop(["Outlet_Size","Outlet_Location_Type","Outlet_Type","Item_Fat_Content"],axis=1,inplace=True)
df.head(5)

y= df.iloc[:,4].values
r=df
r.drop(["Item_Outlet_Sales"],axis=1,inplace=True)
x=r.iloc[:,:].values


#APPLYING RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
cf= RandomForestRegressor(n_estimators=500,criterion="mse",random_state=0)
#cf.fit()
params = {
    
    "n_estimators" : [100,200,300,400,460,600],
    "max_depth" : [3,4,6,8,10,12,16],
    "criterion" : ['mse','accuracy'],
    "bootstrap" : [True,False]
    }

from sklearn.model_selection import RandomizedSearchCV
random_c =  RandomizedSearchCV(cf, param_distributions = params, n_iter=40, verbose=2,random_state=6,n_jobs=-1 ,cv=3)
random_c.fit(x,y)

random_c.best_estimator_

cf =RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=8, max_features='auto', max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=600, n_jobs=None, oob_score=False,
                      random_state=0, verbose=0, warm_start=False)
from sklearn.model_selection import cross_val_score
score = cross_val_score(cf, x, y, cv=6,scoring="MSE")
print(score)
np.sqrt(score.mean)

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
xt,xtest,yt,ytest = train_test_split(x,y,test_size=0.3)

cf.fit(xt,yt)
ypred=cf.predict(xtest)

mse= mean_squared_error(ytest,ypred)

rmse = np.sqrt(mse)

print(rmse)
