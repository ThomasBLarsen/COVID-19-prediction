#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 10:29:21 2020

@author: thomas
"""


import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit





#%% Collect all the daily csv reports and read them to a Dataframe
Folder = '/home/thomas/git/COVID-19/csse_covid_19_data/csse_covid_19_daily_reports'
onlyfiles = [f for f in listdir(Folder) if isfile(join(Folder, f))]

Li = []
for file in onlyfiles:
    if file.split('.')[-1] == 'csv':
        Li.append(pd.read_csv(Folder + '/' + file, index_col=None, header=0))
        
frame = pd.concat(Li, axis=0, ignore_index=True)

frame['ds'] = pd.to_datetime(frame['Last Update'])

#%% Create a new dataframe that groups data by day and Country/Region
Stats = frame.set_index('ds').groupby([pd.Grouper(freq='D'),'Country/Region']).sum()
UniqueCountries = frame['Country/Region'].unique()

for Country in UniqueCountries:
    plt.plot(Stats.xs(Country,level=1, drop_level=True)['Confirmed'])


#plt.legend(UniqueCountries)
plt.show()

#%% Do prediction


#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
#from sklearn.metrics import classification_report, confusion_matrix

Country = 'Mainland China'
Predict = Stats.xs(Country,level=1, drop_level=True)
Predict = Predict.rename(columns={'Confirmed':'y'})
Predict['ds'] = Predict.index - Predict.index[0]

#model = LinearRegression()
#model.fit(np.array(Predict['ds']).reshape(-1, 1), np.array(Predict['y']))

#print('Model Score: ', model.score(np.array(Predict['ds']).reshape(-1, 1), np.array(Predict['y'])))


DataDays = 1
Dates = Predict.index[:-DataDays]
XData = np.array(Predict['ds'].dt.days)[:-DataDays] 
YData = np.array(Predict['y'])[:-DataDays]

def fsigmoid(x, a, b, c):
    return c / (1.0 + np.exp(-a*(x-b)))

p0 = [ 0.2, 8, 7]
popt, pcov = curve_fit(fsigmoid, XData, YData, p0, method= 'trf', bounds=((0.18,1,0),(0.6,50,100000)))
residuals = YData- fsigmoid(XData, *popt)
ss_res = np.sum(residuals**2)

print(popt)

ModelPredict = fsigmoid(XData, *popt)

Futuredays = np.arange(XData[-1],XData[-1] + 60,1)

FuturePredic = fsigmoid(Futuredays, *popt)

FutureDates = []
for Day in Futuredays:
    FutureDates.append(Dates[0] + pd.Timedelta(Day, unit='d'))

plt.Figure(figsize=[16,12])
plt.plot(Dates, YData, linestyle = '-')
plt.plot(Dates,ModelPredict, linestyle = '-')
plt.plot(FutureDates,FuturePredic, linestyle = '--')
plt.legend(['COVID-19 Cases - ' + Country,' Sigmoid fit', 'Predicted'])
plt.grid()
plt.xticks(rotation=30)
plt.show()
