# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 14:38:24 2022

@author: safak
"""
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,ShuffleSplit,GridSearchCV,cross_val_score,cross_val_predict
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.preprocessing import RobustScaler
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
import os
import time
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

df=pd.read_csv("train_new.csv")

l=[]

"""for i in df["Billing Date"]:
    date_str = i
    time_tuple = time.strptime(date_str, "%Y-%m-%d")
    timestamp = time.mktime(time_tuple)
    l.append(timestamp)

df.drop(["Billing Date"],axis=1,inplace=True)
df["Billing Date"]=l"""

df.drop(["Unnamed: 0"],axis=1,inplace=True)
df.drop(["Billing Date"],axis=1,inplace=True)

x=df.drop(["Litres"],axis=1)
y=df["Litres"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

x_train=x_train.reset_index()
x_test=x_test.reset_index()
y_train=y_train.reset_index()
y_test=y_test.reset_index()

x_train.drop(["index"],axis=1,inplace=True)
x_test.drop(["index"],axis=1,inplace=True)
y_train.drop(["index"],axis=1,inplace=True)
y_test.drop(["index"],axis=1,inplace=True)

scaler=StandardScaler()
scaler.fit(x_train)

x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)
#scaler=RobustScaler()
#scaler.fit(x_train)

x_train_scaled=scaler.transform(x_train)
x_test_scaled=scaler.transform(x_test)

#MAKİNE ÖĞRENMESİ MODELİ (LINEAR REGRESSION)
lm=LinearRegression()
model=lm.fit(x_train,y_train)

print("hata skoru: {}".format(np.sqrt(-cross_val_score(model,x_test,y_test,cv=10,scoring="neg_mean_squared_error")).mean()))

val=0
print("        TAHMİN    -    GERCEK\n")
for i in range(100):
    val=model.predict(x_test.iloc[i:i+1,:])
    print("{} - {}".format(val[0][0],y_test.iloc[i:i+1,:]["Litres"].values[0]))
    
predictions=[]
for i in range(len(x_test)):
    val=model.predict(x_test.iloc[i:i+1,:])
    predictions.append(val[0])

print(np.sqrt(mean_squared_log_error( y_test.values, predictions )))




#1.005494255000526 -> with billing date (convert to float)
#1.0057759498730925-> without billing date (convert to float)























































