"""
Created on Mon Nov 14 16:38:57 2022

@author: micha
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split


import os
cwd = os.getcwd()
print(cwd)
os.chdir("C:\\Users\micha\OneDrive - Dundalk Institute of Technology\Year Three\DataScience\CA2")
car_data= pd.read_csv("cleanedData.csv")

# =============================================================================
#-----NOTE 
# Originally I had planned to do my predictions based off of the price of the car
# However upon conducting my search it was prediciting an astounding 1% with a 9000k 
# mean error, so i updated it to instead predict the distance a car has travelled
# =============================================================================

#############################FUNCTIONS########################################
#function for calculating the rsquare and rsqaure adjusted
#takes in the x_train, y_train and model
#created this function as i realised it was the same code repeated x number of times
def modelTraining(x_train, y_train, model):
    raw_sum_sq_errors = sum((y_train.mean() - y_train)**2)
    prediction_sum_sq_errors = sum((predictions_train - y_train)**2)
    rsquared1 = 1-prediction_sum_sq_errors/raw_sum_sq_errors
    
    N= 431 #len(y_train)
    p=1 
    rsquared_adj = 1 - (1-rsquared1)*(N-1)/(N-p-1)
    print("Rsquared Regression Model: " +str(rsquared1))
    print("Rsquared Adjusted Regression Model: " +str(rsquared_adj))

def meanAverageError(predictions_test, y_test):
    print("meanAverageError")
    mae = sum(abs(predictions_test - y_test))/(len(y_test))
    #Prediction_test_MAE = sum(abs(predictions_test - y_test))/len(y_test)
    return mae

#function that calculates the mean average percentage error by taking in the predicited test
# and the y_test
def meanAveragePercentageError(predictions_test, y_test):
    print("meanAveragePercentageError")
    mape = np.mean(abs((y_test - predictions_test)/y_test))/(len(y_test))
    #mape = np.mean(np.abs((predictions_test- y_test)/y_test))*100
    return mape

def rmse(predictions_test, y_test):
    print("rmse")
    rmse = (sum((predictions_test - y_test)**2)/len(y_test))**0.5
    return rmse
######################################################################################



##########################Feature Engineering##################################
car_data.info()
car_data.head()
car_data.describe()

# =============================================================================
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   Index       1929 non-null   int64  -- Numerical --- Predictor
#  1   Name        1158 non-null   object -- Categorical --- Predictor
#  2   Engine      1158 non-null   float64 -- Numerical --- Predictor
#  3   Fuel Type   1158 non-null   object  -- Categorical --- Predictor
#  4   KM          1158 non-null   float64 -- Numerical --- Response
#  5   Year        1158 non-null   float64 -- Numerical --- Predictor
#  6   Price       1158 non-null   float64 -- Numerical --- Predictor
#  7   Location    1158 non-null   object -- Categorical --- Predictor
# =============================================================================
car_data.isnull().sum()


#########Feature Engineering Step 2: Drop certain variables if not required

# Drop Unnamed as unique values
print(len(car_data.Index.unique())) 
car_data.drop('Index', axis = 1, inplace = True) 

#Dropped Name as it isnt relevant 
car_data.drop('Name', axis=1, inplace =True)

#########Feature Engineering Step 3: Construct New Variables if required
#Change Fuel to numerical
car_data['Fuel']=np.where(car_data.FuelType =="Petrol",1,0)
car_data.drop('FuelType', axis=1, inplace=True)
car_data['CarsFromDublin']=np.where(car_data.Location =="Dublin",1,0)
car_data['CarsFromCork']=np.where(car_data.Location =="Cork",1,0)
car_data['CarsFromWaterford']=np.where(car_data.Location =="Waterford",1,0)
car_data.drop('Location', axis=1, inplace = True)

car_data.info()
# =============================================================================
# No Strong Correlation so no need for Multicolinearity
# =============================================================================

#Produce scatter and correlation plots
#Shows there is no strong correlation between price and any other variables
corrVals = car_data.corr()
print(corrVals)

#Plot of relationships between different car variables
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.pairplot(car_data)

#Heatmap to show the correlation between the variables
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.heatmap(car_data.corr(), annot=True, cmap = 'Greens')
plt.show()

#Box plot for the newly added variables
#Fuel Types and the distances they travelled
#Shows that Diesel cars on average travelled more then petrol cars with their IQR
# ranging from 12-18000 whereas petrol cars are only 5 -13000km travelled travelled
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(car_data.Fuel, car_data.KM)
plt.title("Comparing Petrol Vs Diesel")
plt.show()

#CarLocation of the three most common places and the distances travelled
#Shows cars outside Dublin have travelled longer distances then cars within dublin
#not surprising considered how frequent public transport is 
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(car_data.CarsFromDublin, car_data.KM)
plt.title("Comparing the Distance Travelled of Cars from Dublin vs Other")
plt.show()

#Cars from Cork have an IQR of 4-18000km but also have a maxiumum distance of 18000km
#whereas cars outside cork have a larger maximum distance of 30000km
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(car_data.CarsFromCork, car_data.KM)
plt.title("Comparing the Distance Travelled of Cars from Cork vs Other")
plt.show()

#Cars from Waterford are dwarfed in terms of distance travelled when compared to
#cars outside of the county, they have a very small IQR ranging from 5-9000km with 9000km
#being the max range as well
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(car_data.CarsFromWaterford, car_data.KM)
plt.title("Comparing the Distance Travelled of Cars from Waterford vs Other")
plt.show()

car_data.info()
###############REGRESSION MODELLING######################################
#Step One:  Split the Data
#Split the data between predictor and regression
x = car_data[['Engine', 'Price', 'Year', 'Fuel', 'CarsFromDublin', 'CarsFromCork', 'CarsFromWaterford']]
y = car_data['KM'] 

#Splitting the Data Set into Training Data and Test Data
#Splitting it 67/33 train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.333)
y_train 
x_train 

#Step Two:  Model Selection
#Fit the variables in order of strongest correlation with KM and calculate adjusted R squared at each step.
#NOTE: Orginally had this as price but updated to KM due to a better correlation
#example the strongest correlation for price was cars from cork which had 0.036

# =============================================================================
# List of Correlation from Strongest to Weakest
# Year ---- Strongest ----- -0.81
# Fuel
# Cars From Waterford
# Engine
# Cars From Dublin
# Cars From Cork
# Price ---- Weakest ----- 0.0031
# =============================================================================
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model2 = LinearRegression()
model3 = LinearRegression()
model4 = LinearRegression()
model5 = LinearRegression()
model6 = LinearRegression()
model7 = LinearRegression()

print("")
#########################Model1###############################
print("Model One: Year")
#Adding Year to the list
model1.fit(x_train[['Year']], y_train)

#the coef is used to describe indicate the relationship of the variable
#A positive sign indicates that as the predictor variable increases, 
#the response variable decreases
#A negative sign indicates that as the predictor variable increases,
#the response variable decreases.
print("modelOne coef:" , str(model1.coef_))
#the expected mean value of Y when all X=0. 
print("modelOne intercept", str(model1.intercept_))
#In order to predict the distance travelled you take the intercept value shown above
# and add it to the coef times by the train value
#The Year value is negative which means the newer car will have less distance travelled then
#an older car
#KM = -13711.97983325* Year + 27757644.55447659 



#Generate predictions for the train data
predictions_train = model1.predict(x_train[['Year']])

models =[]
models.append(modelTraining(x_train,y_train, model1))
print("")
# =============================================================================
# Rsquared Regression Model: 0.651396120409319 = 66%
# Rsquared Adjusted Regression Model: 0.6505835239533967 =65%
# =============================================================================
#########################Model2###############################
print("Model Two: Year & Fuel")
#Adding Fuel to the model
model2.fit(x_train[['Year', 'Fuel']], y_train)
print(model2.coef_)
print(model2.intercept_)
#KM = 27757644.55447659*Year + -31460.50349713*Fuel + 25809303.114316154

predictions_train = model2.predict(x_train[['Year','Fuel']])

models2 =[]
models2.append(modelTraining(x_train,y_train, model2))
print("")
# =============================================================================
# Rsquared Regression Model: 0.7057400497421817 = 71%
# Rsquared Adjusted Regression Model: 0.7050541291122101 = 71%
# =============================================================================
#########################Model3###############################
print("Model Three: Year, Fuel & Cars from Waterford")
#Adding Cars From Waterford
model3.fit(x_train[['Year', 'Fuel', 'CarsFromWaterford']], y_train)
print(model3.coef_)
print(model3.intercept_)
#KM = -12679.44625052*Year + -30912.6078414*Fuel + -3114.48760177*CarsFromWaterFord+ + 25809303.114316154


predictions_train = model3.predict(x_train[['Year','Fuel', 'CarsFromWaterford']])

models3 =[]
models3.append(modelTraining(x_train,y_train, model3))
print("")
# =============================================================================
# Rsquared Regression Model: 0.7065682232368578 = 71%
# Rsquared Adjusted Regression Model: 0.7058842330812327 = 71%
# =============================================================================
#########################Model4###############################
print("Model Four: Year, Fuel, CarsFromWaterford & Engine")
#Adding Engine
model4.fit(x_train[['Year', 'Fuel', 'CarsFromWaterford', 'Engine']], y_train)
print(model4.coef_)
print(model4.intercept_)
#KM = -12615.67263087*Year + -47356.38403073*Fuel + -10005.13369497*CarsFromWaterFord+ -41033.06957118*Engine + 25622359.89234949

predictions_train = model4.predict(x_train[['Year','Fuel', 'CarsFromWaterford', 'Engine']])

models4 =[]
models4.append(modelTraining(x_train,y_train, model4))
print("")
# =============================================================================
# Rsquared Regression Model: 0.7131489824132342  = 71%
# Rsquared Adjusted Regression Model: 0.7124803320225891 = 71%
# =============================================================================
#########################Model5###############################
print("Model Five: Year, Fuel, CarsFromWaterford ,Engine & CarsFromDublin")
#Adding CarsFromDublin
model5.fit(x_train[['Year', 'Fuel', 'CarsFromWaterford', 'Engine', 'CarsFromDublin']], y_train)
print(model5.coef_)
print(model5.intercept_)
#KM = -12657.08675659*Year + -44161.59586249*Fuel + -6472.06942659*CarsFromWaterFord+ -34827.96554519*Engine +  10928.65236702*CarsFromDublin+ 25693263.138610736

predictions_train = model5.predict(x_train[['Year','Fuel', 'CarsFromWaterford', 'Engine', 'CarsFromDublin']])
 
models5 =[]
models5.append(modelTraining(x_train,y_train, model5))
print("")
# =============================================================================
# Rsquared Regression Model: 0.7180111528492672 = 72%
# Rsquared Adjusted Regression Model: 0.7173538361892422 = 72%
# =============================================================================
#########################Model6###############################
print("Model Six: Year, Fuel, CarsFromWaterford ,Engine, CarsFromDublin & CarsFromCork ")
#Adding Cars From Cork
model6.fit(x_train[['Year', 'Fuel', 'CarsFromWaterford', 'Engine', 'CarsFromDublin', 'CarsFromCork']], y_train)
print(model6.coef_)
print(model6.intercept_)
#KM = -12600.67338118*Year + -45398.82843539*Fuel + -7390.22010528*CarsFromWaterFord+ -36350.50121446*Engine +   10022.66878778*CarsFromDublin + -4994.13462423*CarsFromCork+ 25582857.487180654


predictions_train = model6.predict(x_train[['Year','Fuel', 'CarsFromWaterford', 'Engine', 'CarsFromDublin', 'CarsFromCork']])

model6 =[]
model6.append(modelTraining(x_train,y_train, model6))
print("")
# =============================================================================
# Rsquared Regression Model: 0.718143997053015 = 72%
# Rsquared Adjusted Regression Model: 0.7174869900531387 =72%
# =============================================================================
#########################Model7###############################
print("Model Seven: Year, Fuel, CarsFromWaterford ,Engine, CarsFromDublin ,CarsFromCork & Price")

#Adding Price
model7.fit(x_train[['Year', 'Fuel', 'CarsFromWaterford', 'Engine', 'CarsFromDublin', 'CarsFromCork', 'Price']], y_train)
print(model7.coef_)
print(model7.intercept_)
#KM = -1.26133024e+04*Year + -4.53991288e+04*Fuel + -7.07015723e+03*CarsFromWaterFord+ 3.62781210e+04*Engine +   1.00202064e+04*CarsFromDublin + -4.81874893e+03*CarsFromCork+  -9.28771300e-02*Price+ 25609662.63067534

predictions_train = model7.predict(x_train[['Year','Fuel', 'CarsFromWaterford', 'Engine', 'CarsFromDublin', 'CarsFromCork', 'Price']])
models =[]
models.append(modelTraining(x_train,y_train, model7))

#Displaying coefficients by placing them in a DataFrame.
output = pd.DataFrame(model7.coef_, ['Year COEFF', 'Fuel COEFF', 'CarsFromWaterford COEFF', 'Engine COEFF', 'CarsFromDublin COEFF', 'CarsFromCork COEFF', 'Price COEFF'], columns = ['Coeff'])
print(output)  
# =============================================================================
# Rsquared Regression Model: 0.7182429375528141 = 72%
# Rsquared Adjusted Regression Model: 0.7175861611834733 = 72%
# =============================================================================
#Coeff for the final model
# =============================================================================
#                                 Coeff
# Year COEFF              -12958.666908 ---- As the Year(when a car is made)Increases the KM decreases
# Fuel COEFF              -41601.173508 ---- Not sure how to desribe this one
# CarsFromWaterford COEFF  -5537.422575 ---- The more cars from Waterford KM descreases 
# Engine COEFF            -29953.244097 ---- As the Engine size icreases the KM decreases
# CarsFromDublin COEFF     12538.069636 ---- The more cars from Dublin KM increases
# CarsFromCork COEFF        2508.440712 ---- The more cars from Cork KM increases
# Price COEFF                  0.223042 ---- Does not have an effect as its near 0?
# =============================================================================
# Interesting to plot the errors for the actual values
plt.scatter(y_train, predictions_train)
p1 = max(max(predictions_train), max(y_train))
p2 = min(min(predictions_train), min(y_train))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.show()  # Should be close to a straight line

plt.scatter(y_train, predictions_train - y_train)
plt.show()


#Step 3: Model Evaluation Based on TEST set.
#Based of the results of the previous Step, I will be using model7 as model5, model6 and 
#model7 had an adjusted Rsquare of 72%

#MAE - Mean Absolute Error
#MAPE - Mean Absolute Percentage Error
#RMSE - Root Mean Square Error



### For some reason the results of the MAPE & RMSE are incorrect,
### I have tried googling solutions for this as well as reviewing previous 
### Class examples and I am unable to see a solution for this

predictions_test = model5.predict(x_test[['Year','Fuel', 'CarsFromWaterford', 'Engine', 'CarsFromDublin']])
print(len(y_test))
# =============================================================================


print(meanAverageError(predictions_test, y_test))
print(meanAveragePercentageError(predictions_test, y_test))
#print(rmse(predictions_test, y_test))
# =============================================================================





# =============================================================================
###Plot prediction results
#First plot the y test values and the predictions for the model
#This SHOULD BE close to a straight line
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test)
p1 = max(max(predictions_test), max(y_test))
p2 = min(min(predictions_test), min(y_test))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.title("Predictions v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Predicted Values")
plt.show() 
#Looking at this graph you can see that some of the points sit on the fitted line
#with many of these plots sitting just under or slightly over suggesting the predicted
#values are nearly accurate to the actual values and in some cases spot on

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
plt.scatter(y_test, predictions_test - y_test)
plt.title("Errors v Actual Test Values")
plt.xlabel("Actual values")
plt.ylabel("Error Values")
plt.show()
#This Graph shows that there is no real pattern between Error Values and Actual Values
#with many of them being scattered throughout this graph, However it seems that the
#larger KM distance then the error value predicted was less then the actual value 
#and many of the smaller KM distance error values were over the actual value 
# =============================================================================

