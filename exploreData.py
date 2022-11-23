# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 12:34:28 2022

@author: micha
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


# =============================================================================
# Importing the Data from the data.csv that was created in storedDataFinal
# =============================================================================
import os
cwd = os.getcwd()
print(cwd)
os.chdir("C:\\Users\micha\OneDrive - Dundalk Institute of Technology\Year Three\DataScience\CA2")

data= pd.read_csv("data.csv")

#Who this benefits - Most Importantly Me, as I am looking to buy a new car in Jan
# Benefits people selling cars as they will be able to compare their price with others
#Car Dealerships see if they are selling cheaper then a private seller


# STEP ONE- IDENTIFYING VARIABALES
# =============================================================================
data.info()
#  0   Name      1113 non-null   object - Categorical
#  1   Engine    1113 non-null   object - Categorical
#  2   KM        1113 non-null   int64  - Numerical
#  3   Year      1113 non-null   int64  - Numerical
#  4   Price     1113 non-null   int64  - Numerical
#  5   Location  1113 non-null   object - Categorical
# =============================================================================

#STEP TWO - CLEANING DATA

#Cleaning the KM column and removing any records under 0
print(data.KM.min())
print(data.KM.unique())
data = data.drop(data[data.KM<0].index)

print(data.KM.unique())
numberKM = data.KM.value_counts()
print(numberKM)

#Cleaning the Year Column and removing any under 0
print(data.Year.min())
print(data.Year.unique())
data = data.drop(data[data.Year<0].index)

print(data.Year.unique())
numberYear = data.Year.value_counts()
print(numberYear)

#Cleaning the Price Column and removing any under 0
print(data.Price.min())
print(data.Price.unique())
data = data.drop(data[data.Price<0].index)

print(data.Price.unique())
numberPrice = data.Year.value_counts()
print(numberPrice)

#Cleaning Location
print(data.Location.unique())

data.info()
            
#STEP 3 - MISSING VALUES
# Checking the missing values
# =============================================================================
data.isnull().sum()
# Name        0
# Engine      0
# KM          0
# Year        0
# Price       0
# Location    0
# =============================================================================

#STEP 4 - OUTLIERS
# =============================================================================
# #NOTE UNABLE TO RUN THE SNS IN SPYDER, BUT CAN UNCOMMENT AND RUN IN PYCHARM
# =============================================================================

#EXAMINE YEAR
data.Year.describe()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.Year)
plt.title("Year Boxplot")
plt.show()

# =============================================================================
# Year Box Plot: IQR for the year the car was manufactured is 
# around the 2014 to 2018 with the median of these cars being 2016, 
# the minimum is roughly 2007 and the max is the current year, 
# for this box plot there is no outlier beyond the min and max, 
# (which was expected for the max as it’s the current 
#  year but a bit surprising for the minimum as a car 
#  could be older then 2007
# =============================================================================


#EXAMINE KM
data.KM.describe()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.KM)
plt.title("KM BoxPlot")
plt.show()

# =============================================================================
# The IQR for the KM box is between 5000 and 15000km with 
# the median being around 12500km, the max for the km is 
# around 30000km and the minimum is around 0, which isn’t 
# surprising as many of the models being sold from a dealership 
# maybe brand new and would have less then 10km based off how far 
# they are driven to the dealership.
# (There are no Outliers here, as I removed any km less then 0 a 
#  previous stage)
# =============================================================================

#EXAMINE PRICE
data.Price.describe()
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
sns.boxplot(x=data.Price)
plt.title("Price BoxPlot")
plt.show()

# =============================================================================
# The IQR for price is low between nearly 10k and 20k with 
# the median price being 15k. the minimum price based off 
# the boxplot is around 1k maybe slightly less and the max 
# is around 35k. There are a few outliers out past the maximum 
# suggesting there are higher end cars, brand new cars 
# (or perhaps people value the cars higher) in this dataset
# =============================================================================

#5. EXPLORATORY DATA ANALYSIS - UNIVARIATE ANALYSIS
# =============================================================================
# 
# Name        Categorical- but no Analysis needed
# Engine      Categorical- find the amount of Petrol, Diesel
# KM          Numerical - Min, Max, Median
# Year        Numerical - Min, Max, Median
# Price       Numerical - Min, Max, Median
# Location    Categorical - find the amount of different locations

#No BoxPlots in this section as they are all above in the outliers section
# =============================================================================

#Engine
# =============================================================================
print(data.Engine.unique())
# #FreqTable
engineSizes = data.Engine.value_counts()
print(engineSizes)
#1.5 Diesel      954 --- Most Common
#1.0 Petrol      257
#1.1 Petrol      240
#1.3 Petrol       68
#1.2 Petrol       56
#1.6 Diesel       54
#2.0 Petrol       54
#2.0 Diesel       54
#1.8 Diesel       54
#1.9 Diesel       54
#0.9 Petrol       54
#1.6 Petrol       16
#2.3 Diesel        8
#0.0 Electric      6 --- Least Common
# =============================================================================

#autopct='%1.1f%%', used to display percentage however
#piechart looked very messy with it and was hard to read
engineSizes.plot.pie(autopct='%1.1f%%', wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
plt.title("Unique Engines")
plt.show()

#As the Piechart is a bit messy, I also added a barchart as well
engineSizes.plot.bar(color = "green", width = 0.5)
plt.title("Unique Engines")
plt.xlabel("Engine Sizes")
plt.ylabel("Number of Cars")
plt.show()

#1.5 litre diesel cars were the most common sold with over 50% 
#off all the cars being the 1.5, the second largest was a 1.1 litre
#petrol with 9%

#KM
# =============================================================================
data.KM.describe()
print(data.KM.unique())
# =============================================================================

plt.hist(x=data.KM, color="red", edgecolor="black" )
plt.title("Distance Cars Have Travelled")
plt.xlabel("KM")
plt.ylabel("Number of Cars")
plt.show()


####################Alternative Engine
#data['Fuel']=np.where(data.FuelType =="Petrol",1,0)
#piechartlabels = ["Diesel", "Petrol"]
#engineSizes = data.Fuel.value_counts()
#engineSizes.plot.pie(autopct='%1.1f%%', wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'}, labels = piechartlabels)

#plt.title("Unique Engines")
#plt.show()

#YEAR
# =============================================================================
data.Year.describe()
print(data.Year.unique())
# =============================================================================

plt.hist(x=data.Year, color="green", edgecolor="red")
plt.title("The Years of Cars Sold")
plt.xlabel("Year")
plt.ylabel("Number of Cars")
plt.show()


#PRICE
# =============================================================================
data.Price.describe()
print(data.Price.unique())
# =============================================================================

plt.hist(x=data.Price, color="skyblue", edgecolor="black")
plt.title("The Price of Cars")
plt.xlabel("Price")
plt.ylabel("Number of Cars")
plt.show()

#Location
# =============================================================================
print(data.Location.unique())
# FreqTable
uniqueLocations = data.Location.value_counts()
print(uniqueLocations)
#Dublin       242
#Waterford    158
#Limerick     114
#Cork         114
#Kerry        104
#Carlow        93
#Kildare       63
#Meath         57
#Clare         55
#Kilkenny      53
#Louth         53
#Sligo         53
#Galway        51
#Wicklow       14
#Laois          3
# =============================================================================

#autopct='%1.1f%%', used to display percentage however
#piechart looked very messy with it and was hard to read
uniqueLocations.plot.pie(wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'})
plt.title("Locations where cars are being sold")
plt.show()
#Looking at the chart you can see that dublin had over 19.7% 
#of carts with waterford the second highest at 12.9% cork and
#limerick were tied at 9.3% each

#Similar to Engines, piecharts a bit messy, so barchart for 
#tidiness
uniqueLocations.plot.bar(color = "hotpink", width=.8)
plt.title("Locations where cars are being sold")
plt.xlabel("Location Names")
plt.ylabel("Number of Cars")
plt.show()

#6. EXPLORATORY DATA ANALYSIS - BIVARIATE ANALYSIS
# =============================================================================
#Engine & Price -  Numerical - Categorical
#KM & Year  -  Numerical - Numerical
#Price & Location - Numerical - Categorical
#Year & Price Numerical - Numerical
#KM & Location - Numerical - Categorical
# =============================================================================

#Engine & Price – Question One:  Do Cars with a Bigger Engine Cost more money then a Car with a smaller One?
#First I will find the mean and median for each categorical value
data.groupby('Engine')['Price'].mean()
data.groupby('Engine')['Price'].median()
#At first glance it does not appear that the size the engine effect 
#the price of a car
#Show the two on a boxplot
sns.boxplot(data.Engine, data.Price)
plt.show()
#The box shows far too many outliers for each engine size
#Conclusion I don't think the engine size has an effect on the price

#KM & Years – Question Two: Do Older cars have a higher mileage then cars manufactured after them
#First I will use a scatter graph to see if there is a postive correlation
#between the two
plt.scatter(data.Year, data.KM)
plt.plot()
plt.xlabel('Year')
plt.ylabel('KM')
plt.show()
data[['Year','KM']].corr()

#Looking at the scatter graph we can there 
#is a strong negative correlation between 
#the higher mileage cars and the year produced, 
#this graph shows that the more newer cars have 
#less mileage then the older one(which was expected) 
#the correlation for the was -0.8
# =============================================================================

#Price and Location – Question 3: Are Cars from the "richer parts" of Ireland 
#more expensive to purchase then cars from “poorer”  counties 

#https://twitter.com/r_o_farrell/status/1493912709800341507 - states that 
#Dublin, Limerick, Kildare, Meath, Wicklow & Cork are richest

#the price average price for counties as followed
data.groupby('Location')['Price'].mean()
# =============================================================================
#Carlow       14978.172043
#Clare        14453.709091
#Cork         15184.245614 *
#Dublin       15341.219008 *
#Galway       16406.529412
#Kerry        14925.394231
#Kildare      13651.269841 *
#Kilkenny     13860.509434
#Laois         8746.333333
#Limerick     13818.394737 *
#Louth        15893.924528
#Meath        15610.473684 *
#Sligo        15693.547170
#Waterford    14717.303797
#Wicklow      12555.000000 *
# =============================================================================

data.groupby('Location')['Price'].median()
#sns.boxplot(data.Location, data.Price)
plt.show()
priceLocPivot = data.pivot_table(index=['Location'], values=['Price'], aggfunc='sum')
print(priceLocPivot)

# =============================================================================
# Looking at the average prices of cars from each county 
# in the dataset, you can see that the prices are pretty 
# close together 4 counties have a higher price average then 
# Dublin (Galway, Louth, Meath & Sligo) with only one of these 
# counties appearing on the rich list. In face Limerick which was 
# ranked second in the list was the 5th cheapest place to get a car 
# with Wicklow being the 2nd lowest on the list. These results show 
# that maybe counties that are perceived as better off do not value 
# cars as much as the other counties 
# =============================================================================

#Year & Price – Question Four: Are cars that are newer more likely to be more or less expensive 
plt.scatter(data.Year, data.Price)
plt.show()
data[['Year','Price']].corr()
# =============================================================================
#           Year     Price
#Year   1.000000 -0.021226
#Price -0.021226  1.000000
# =============================================================================
#This shows there is no strong correlation between the two,which surprised me However there
#are numerous reasons for this such as the size of the datasheet being used, as these
#are second hand cars the owners could be overpricing their cars and therefore throwing the
#correlation off

#Location & KM - Are cars from counties that have a higher urban population such as Dublin
#more or less likely to have have cars with lower distanve travelled
# =============================================================================
data.groupby('Location')['KM'].mean()
# Carlow       113239.612903
# Clare        126167.490909
# Cork         105420.868421 *
# Dublin       131295.723140 *
# Galway       130548.000000 *
# Kerry        165894.230769
# Kildare       10705.984127
# Kilkenny     201772.000000
# Laois          7599.000000
# Limerick     104791.000000 *
# Louth        122103.000000
# Meath        105216.701754
# Sligo        181000.000000
# Waterford     65141.240506 *
# Wicklow      107362.857143
# =============================================================================


data.groupby('Location')['KM'].median()

sns.boxplot(data.Location, data.KM)
plt.show()
pivot = data.pivot_table(index=['Location'], values=['KM'], aggfunc='sum')
print(pivot)

#Looking at this list (the 5 counties with cities that have the highest 
#population have *) we can see that out of the 5, 3 of the counties 
#(Cork, Limerick and Waterford are in the bottom 5.)Dublin and Galway 
#are the exception as the make up the top 5 counties with the highest km 
#travelled however they are a fair distance behind the 3rd placed Kerry.
#These results show that there is less need for cars in this well built area 
#probably due to closeness of work and/or public transport

#data.to_csv('cleanedData.csv')
