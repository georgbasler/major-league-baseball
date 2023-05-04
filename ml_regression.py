#!/usr/bin/env python
# coding: utf-8

# # ML Regression



import pandas as pd
import numpy as np
df = pd.read_csv("mlb_weight_ht.csv")
df.head()


df.isnull().values.any()


df = df.dropna()
df.isnull().values.any()


df.rename(index=str, 
             columns={"Height(inches)": "Height", "Weight(pounds)": "Weight"},
             inplace=True)




from sklearn import linear_model
from sklearn.model_selection import train_test_split



var = df['Weight'].values
var.shape



y = df['Weight'].values #Target
y = y.reshape(-1, 1)
X = df['Height'].values #Feature(s)
X = X.reshape(-1,1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)



lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)




from matplotlib import pyplot as plt
plt.scatter(y_test, predictions)
xline = [min(y_test), max(predictions)]
yline = [min(y_test), max(predictions)]
plt.plot(xline, yline, color='red')
plt.xlabel("Actual Weight")
plt.ylabel("Predicted Weight")



print("Coefficient of determination (R^2): " + str(model.score(X_test, y_test)))




# encode Team and Position categories as booleans
teams = pd.unique(df['Team'])
positions = pd.unique(df['Position'])
dummies = pd.get_dummies(df[['Position', 'Team']])
df2 = pd.concat([df, dummies], axis=1)
df2.drop(columns=['Team', 'Position'], inplace=True)
# add all meaningful features
y = df['Weight'].values #Target
y = y.reshape(-1, 1)
X = df2[dummies.columns].values
#X = X.reshape(-1,1)
# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print("Coefficient of determination (R^2): " + str(model.score(X_test, y_test)))


