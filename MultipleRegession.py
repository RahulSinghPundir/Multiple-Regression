from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reading csv file for which we have to train model
df=pd.read_csv('MachineLearningWithPython\Linear Regression\MultipleRegression\FuelConsumptionCo2.csv')
#print(df.head())# This will give an idea that how our dataset is looking

#Creating a random choice so we do not suffer from bias or train and test should be fair  
msk=np.random.rand(len(df))<0.8 #Giving a 80% of data to train
train,test=df[msk],df[~msk] #Giving a 20% of data to test

# print(train.shape,test.shape)# confirm the size of train and test data set

#Creating a train_x,train_y,test_x and test_y for Multiple Regression model
train_x,train_y=train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']],train[['CO2EMISSIONS']]
test_x,test_y=test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']],test[['CO2EMISSIONS']]

#Plotting a graph of training set for an over all look 
plt.scatter(train.FUELCONSUMPTION_HWY,train.CO2EMISSIONS,color='red')
plt.scatter(train.FUELCONSUMPTION_CITY,train.CO2EMISSIONS,color='blue')
plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='green')
plt.scatter(train.CYLINDERS,train.CO2EMISSIONS,color='orange')
plt.xlabel('R-fuel_HWY,B-fuel_City,G-Enginesize,O-Cylinders')
plt.ylabel('Co2 Emissions')
plt.show()


# Fiting the Multiple Regression model for choosing best fit line with suitable intercept and coefficent
model=linear_model.LinearRegression()
model.fit(train_x,train_y)
print("Coefficient of Features:",model.coef_)

# Predicting the output for test_x that what our model will preidict for test_x
predict=model.predict(test_x)

#MSE is the cost fuction or error that how much our model is far from the actual output/test_y Mean Squared Error
mse=np.mean((test_y-predict)**2)
print("Mean absolute error: ",*mse)

# Explained variance score: 1 is perfect prediction
print("Score to cheack model accuracy:",model.score(test_x,test_y))

