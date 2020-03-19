#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 10:29:21 2020

@author: thomas
"""


import pandas as pd
import matplotlib.pyplot as plt
from modules import Import_CSSEGISandData
from modules import Prediction


#%% Collect all the daily csv reports and read them to a Dataframe
Folder = '/home/thomas/git/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports'
Dataset = Import_CSSEGISandData.Read_Dataset(Folder)


#%% Create a new dataframe that groups data by day and Country/Region
Stats = Dataset.set_index('ds').groupby([pd.Grouper(freq='D'),'Country/Region']).sum()
UniqueCountries = Dataset['Country/Region'].unique()

for Country in UniqueCountries:
    plt.plot(Stats.xs(Country,level=1, drop_level=True)['Confirmed'])
plt.show()

#%% Calcualte the reformation rate per day for all countries

for Country in UniqueCountries:
    # Filter data for selected country
    Dataseries = Stats.xs(Country,level=1, drop_level=True)
    # Calculate the reproduct rate per day
    Rday = Prediction.RDayCalc(Dataseries, FilterCoeff = 0.2)
    plt.plot(Rday['Rday_Filtered'])
    
plt.grid()
plt.xticks(rotation=30)
plt.show()
#%% Do Logistic fitting

# print(UniqueCountries) # see all countries available
Country = 'Mainland China'

# Fit a Generalised logistic function
Prediction.Generalized_logistic_fit(Stats, Country)

#%%
CountryList = ['Mainland China', 'Norway', 'Denmark','Italy']
for Country in CountryList:
    # Filter data for selected country
    Dataseries = Stats.xs(Country,level=1, drop_level=True)
    # Calculate the reproduct rate per day
    Rday = Prediction.RDayCalc(Dataseries, FilterCoeff = 0.2)
    plt.plot(Rday['Rday_Filtered'])
    
plt.legend(CountryList)
plt.grid()
plt.xticks(rotation=30)
plt.show()




#

