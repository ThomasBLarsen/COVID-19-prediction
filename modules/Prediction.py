#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 22:12:12 2020

@author: thomas
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime as dt


def Generalized_logistic_fit(Stats, Country):
    # Filter and rename some data
    Predict = Stats.xs(Country,level=1, drop_level=True)
    Predict = Predict.rename(columns={'Confirmed':'y'})
    Predict['ds'] = Predict.index - Predict.index[0]
    
    
    DataDays = 1 #Filter n number of days. default to 1, which uses the last data available
    Dates = Predict.index[:-DataDays]
    XData = np.array(Predict['ds'].dt.days)[:-DataDays] 
    YData = np.array(Predict['y'])[:-DataDays]
    
    # Define the modified sigmoid function
    def fsigmoid(x, a, b, c):
        return c / (1.0 + np.exp(-a*(x-b)))
    
    p0 = [ 0.2, 8, 7] # Set some initial guess for the coefficients
    popt, pcov = curve_fit(fsigmoid, XData, YData, p0, method= 'trf', bounds=((0.18,1,0),(0.6,50,100000)))
    residuals = YData- fsigmoid(XData, *popt)
    ss_res = np.sum(residuals**2) # Calculate R² 
    print('R²: ' + str(ss_res))
    
    ModelPredict = fsigmoid(XData, *popt) # Calculate the current data fit
    
    # Create future days to predict
    Futuredays = np.arange(XData[-1],XData[-1] + 60,1)
    FuturePredic = fsigmoid(Futuredays, *popt)
    
    # Generate timestamps for the future days. 
    FutureDates = []
    for Day in Futuredays:
        FutureDates.append(Dates[0] + pd.Timedelta(Day, unit='d'))
        
    # Plot the prediction
    plt.Figure(figsize=[16,12])
    plt.plot(Dates, YData, linestyle = '-')
    plt.plot(Dates,ModelPredict, linestyle = '-')
    plt.plot(FutureDates,FuturePredic, linestyle = '--')
    plt.legend(['COVID-19 Cases - ' + Country,' Sigmoid fit', 'Predicted'])
    plt.grid()
    plt.xticks(rotation=30)
    plt.show()

def RDayCalc(Dataseries, FilterCoeff = 0.6):
    
    Rday = []
    RdayFiltered = []
    Itterator = 0
    for day in Dataseries['Confirmed']:
        if Itterator == 0:
            R = 1
            Rfilter = 1
        else:
            
            if Dataseries['Confirmed'][Itterator-1] == 0:
                R = 0
            else:
                R = float(day)/float(Dataseries['Confirmed'][Itterator-1] )
        
        Rfilter = FilterCoeff * R +  Rfilter * (1-FilterCoeff)   
        Rday.append(R)
        RdayFiltered.append(Rfilter)
        Itterator += 1
    d = {'ys': Dataseries.index, 'Rday': Rday, 'Rday_Filtered': RdayFiltered}
    df = pd.DataFrame(d)
    df = df.set_index('ys')
    return df

   
    
    # RDF = []
    # for Day in Predict.itterows():
    #     yPRev = 
    #     RDF.append([Day[0]['y'], ])
    
    
    