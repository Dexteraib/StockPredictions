#Description: This program predicts the GOOG stock for a specific day 
#             using the Machine Learning algorithm called Support Vector Regression 
#             (SVR) & Linear Regression

#Import necessary libraries 
import pandas as pd 
import numpy as np 
from sklearn.svm import SVR 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

#Create Lists / X and Y datasets 
dates = []
prices = [] 

#Load the data 
from google.colab import files 
uploaded = files.upload()
df = pd.read_csv('goog_predict.csv')
df.head(7)

#Get number of rows and columns in the data set 
df.shape

#Get the last row of data 
df.tail(1)

#Get all of data except for last row  
df = df.head(len(df)-1)
df

#The new shape of the data 
df.shape 

#Gather rows from "Date" column 
df_dates = df.loc[:,'Date']
#Gather rows from "Open" column 
df_open = df.loc[:, 'Open']

#Independent Data set x 
for date in df_dates: 
  dates.append( [int(date.split('-')[2])])
#Independent Data set y 
for open_price in df_open:
  prices.append(float(open_price))

#What dates were recorded 
print(dates)

def predict_prices(dates, prices, x):
  
  #Create the 3 Support Vector Regression models
  svr_lin = SVR(kernel='linear', C= 1e3)
  svr_poly= SVR(kernel='poly', C=1e3, degree=2)
  svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
  
  #Train the SVR models 
  svr_lin.fit(dates,prices)
  svr_poly.fit(dates,prices)
  svr_rbf.fit(dates,prices)
  
  #Create the Linear Regression model
  lin_reg = LinearRegression()
  #Train the Linear Regression model
  lin_reg.fit(dates,prices)
  
  #Plot the models on a graph to see which has the best fit
  plt.scatter(dates, prices, color='black', label='Data')
  plt.plot(dates, svr_rbf.predict(dates), color='red', label='SVR RBF')
  plt.plot(dates, svr_poly.predict(dates), color='blue', label='SVR Poly')
  plt.plot(dates, svr_lin.predict(dates), color='green', label='SVR Linear')
  plt.plot(dates, lin_reg.predict(dates), color='orange', label='Linear Reg')
  plt.xlabel('Days')
  plt.ylabel('Price')
  plt.title('Regression')
  plt.legend()
  plt.show()
  
  return svr_rbf.predict(x)[0], svr_lin.predict(x)[0],svr_poly.predict(x)[0],lin_reg.predict(x)[0]

#Predict the price of GOOG on day 28
predicted_price = predict_prices(dates, prices, [[28]])
print(predicted_price)
