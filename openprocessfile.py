#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 11:35:24 2020

@author: megannguyen
"""
#%% 
#cach 1
#create a file then save in "lesson 3" folder
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
#change directory
#os.chdir("/Users/megannguyen/Desktop/Python/lesson3_open file")
import pandas
df = pandas.read_csv("student-mat.csv")
print(df)
#%%
#cach 2
#import modules csv
import csv

#mo file student-mat.csv de doc du lieu
with open("/Users/megannguyen/Desktop/Python/lesson3_open file/student-mat/student-mat.csv") as f:
    data=csv.reader(f)
#doc tung dong du lieu trong file
    for row in data:
        print(row)
#%%        
#goi ham DictReader() de tra ve cach doc file khac
#reader=csv.DictReader(open("/Users/megannguyen/Desktop/Python/lesson3_open file/student-mat/student-mat.csv"))
#duyet qua tung dong du lieu cua file
#for raw in reader:
#    print(raw)
# %%
import pandas
df = pandas.read_csv("/Users/megannguyen/Desktop/Python/lesson3_open file/student-mat/student-mat.csv")
print(df)
df.head()
# %%
import numpy as np
from sklearn.linear_model import LinearRegression
# %%
df.groupby('failures').mean()
#preprocessing failures into 0 and 1 only
#conditions: if failures > 0 then let failures =1
col         = 'failures' ##define column variable
conditions  = [ df[col] > 0, df[col] == 0 ]
choices     = [ 1,0 ]
    
df["failornot"] = np.select(conditions, choices, default=np.nan)
#%%
x=df[["studytime","freetime","goout"]]
y=df["failornot"]
#%%
model = LinearRegression().fit(x,y)
# %% splitting the dataset into training set and test set
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#print('Training Data Count: {}'.format(x_train.shape[0]))
#print('Testing Data Count: {}'.format(x_test.shape[0]))
# %% 
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse
import seaborn as sns
#pandas.options.display.float_format = '{:.5f}'.format
#x_train = sm.add_constant(x_train)
#results = sm.OLS(y_train, x_train).fit()
#results.summary()
# %%
#Plot actual and estimated scores against each other
import matplotlib.pyplot as plt
#x_test = sm.add_constant(x_test)
#y_preds = results.predict(x_test)
#plt.figure(dpi = 75)
#plt.scatter(y_test, y_preds)
#plt.plot(y_test, y_test, color="red")
#plt.xlabel("Actual Scores")
plt.ylabel("Estimated Scores")
plt.title("Model: Actual vs Estimated Scores")
plt.show()
# %%
#Check errors
#print("Mean Absolute Error (MAE)         : {}".format(mean_absolute_error(y_test, y_preds)))
##mae_sum += abs(y_test - y_pred)
##mae = mae_sum / len(y_test) ##len counts the number of items
#print("Mean Squared Error (MSE) : {}".format(mse(y_test, y_preds)))
##mse_sum += (y_test - y_pred)**2
##mse = mse_sum / len(y_test)
#print("Root Mean Squared Error (RMSE) : {}".format(rmse(y_test, y_preds)))

# %%
#map failornot on studytime

pandas.crosstab(df.studytime,df.failornot).plot(kind='bar')
plt.title('Fail rate for study time')
plt.xlabel('Study Time')
plt.ylabel('Number of Students')

#studytime is a good indicator of fail or not
#%%
#create dummy vars
cat_vars=['studytime','freetime','goout']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pandas.get_dummies(df[var], prefix=var)
    df1=df.join(cat_list)
    df=df1
cat_vars=['studytime','freetime','goout']
data_vars=df.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
#%%
#up sample of not-fail
x = df.loc[:, df1.columns != 'failornot']
y = df.loc[:, df1.columns == 'failornot']
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
columns = x_train.columns
os_data_x,os_data_y=os.fit_sample(x_train, y_train)
os_data_x = pandas.DataFrame(data=os_data_x,columns=columns )
os_data_y= pandas.DataFrame(data=os_data_y,columns=['failornot'])
# we can Check the numbers of our data
print("Length of oversampled data is ",len(os_data_x))
print("Number of no failures in oversampled data",len(os_data_y[os_data_y['failornot']==0]))
print("Number of failures",len(os_data_y[os_data_y['failornot']==1]))
print("Proportion of no failures data in oversampled data is ",len(os_data_y[os_data_y['failornot']==0])/len(os_data_x))
print("Proportion of failures in oversampled data is ",len(os_data_y[os_data_y['failornot']==1])/len(os_data_x))
#%%
#Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model and choose either the best or worst performing feature given smaller and smaller set of features
df_vars=df.columns.values.tolist()
y=['failornot']
x=[i for i in df_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_x, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
#%%
# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(x_train,y_train)

#
y_predlog=logreg.predict(x_test)

print(x_test) #test dataset
print(y_predlog) #predicted values

# %%
#use modules sns and pandas
confusion_matrix = pandas.crosstab(y_test, y_predlog, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot=True)
from sklearn import metrics
print('Accuracy: ',metrics.accuracy_score(y_test, y_predlog))
plt.show()
# %%
