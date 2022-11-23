# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:45:15 2022

@author: micha
"""

from bs4 import BeautifulSoup
import requests	
import pandas as pd


#the cars variable will store all the details 
cars =[]

#All the variables i will be using to store the data, these will be appended
#into cars at the end
carName =[]
carPrice=[]
carEngine=[]
carMileage=[]
carYear=[]
carLocation=[]

#Running a for loop through the first 30 pages on the website
for page in range(1,30):
    #website include the ?search and then the str(page) will increment the page
    html = requests.get("https://www.donedeal.ie/cars/Renault" + "?Search=" +str(page)).text
    soup = BeautifulSoup(html, 'html.parser')
    
    #for loop that runs through the html
    for renaults in soup('div'):
        try:
            #If statement is used to check if its the right div
            if(renaults['class'][0]=="Card__Body-sc-1v41pi0-8"): 
                if(renaults('p')[0]['class'][0]=="Card__Title-sc-1v41pi0-4"):
                    
                    #Getting the Car Name and storing it in the carName variable
                    if(renaults('p')[0]['class'][0]=="Card__Title-sc-1v41pi0-4"):
                        carName.append(renaults('p')[0].contents[0])
                    else:
                        carName.append("Renault")
                        
                    #Getting the carPrice and storing it in the carPame variable
                    if(renaults('p')[2]['class'][0]=="Card__InfoText-sc-1v41pi0-13"):
                        if(renaults('p')[2].contents[0]=='€'):
                            carPrice.append(int(renaults('p')[2].contents[2].replace(',', '')))
                        else:
                            carPrice.append(-1)
                   
                    #looks for the list in the div that contains year, mileage, etc..
                    for ul in renaults('ul'):
                        #the class for the list
                        if ul['class'][0]=="Card__KeyInfoList-sc-1v41pi0-6":
                            
                            if len(ul('li'))>0:
                                
                                #Finding the carEngine
                                if ul('li')[1].contents[0]:
                                    carEngine.append(ul('li')[1].contents[0])
                                else:
                                    carEngine.append("1.0 Petrol")
                                
                                #If statement to check if the cars between no older then 2000
                                if int(ul('li')[0].contents[0])<=2023 and int(ul('li')[0].contents[0])>=2000:
                                    #add this to the carYear list
                                    carYear.append(int(ul('li')[0].contents[0]))
                                else:
                                    carYear.append(2022)
                                
                                if ul('li')[4].contents[0]:
                                    carLocation.append(ul('li')[4].contents[0])
                                else:
                                    carLocation.append("Louth")
                                
                                if ul('li')[2].contents[0].find("km")>=0:
                                    carMileage.append(int(ul('li')[2].contents[0].replace(',', '').replace('km', '').lstrip().rstrip()))
                                else:
                                    carMileage.append(-1)
                                
                                    
                            else:
                                carMileage.append(-1)
                                carYear.append(2022)
                                carLocation.append("Louth")
                                carEngine.append("1.0 Petrol")
                    
                   
            
                    #Attempt to pass the appended data as a dictionary --- Did not work
                    # =============================================================================
    
                    #     details = {
                    #         'CarName':carName,
                    #         'CarEngine':carEngine,
                    #         'CarMileage':carMileage,
                    #         'CarYear':carYear,
                    #         'CarPrice':carPrice,
                    #         }
                    #     cars.append(details)
                    # =============================================================================
        except:   
            pass 


#Merging the variables that were collected into the car list
cars=[[carName[i], carEngine[i], carMileage[i], carYear[i], carPrice[i], carLocation[i]] for i in range(len(carPrice))]
carDataFrame = pd.DataFrame(cars, columns=["Name","Engine", "KM", "Year", "Price", "Location"])


#Exporting the carDataFrame to a csv file
carDataFrame.to_csv("data.csv")
#testing to see if it reads it back in
testData= pd.read_csv("data.csv")


# =============================================================================
# #Taken from GeeksForGeeks ~ possible solution for duplications
# duplicates = testData["CarName"].duplicated(keep = False)
#  
# # bool series
# duplicates
# # passing NOT of bool series to see unique values only
# testData = testData[~duplicates]
# 
# DID NOT WORK :(
# MY SOLUTION WAS TO GO INTO EXCEL AND DELETE ALL DUPLICATES LIKE THAT
# =============================================================================





    
    
