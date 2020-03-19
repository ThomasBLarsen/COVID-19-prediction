#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:52:51 2020

@author: thomas
"""


import pandas as pd
from os import listdir
from os.path import isfile, join
import requests


def Read_Dataset(Folder):
# Read the daily csv reports and read them to a Dataframe

    onlyfiles = [f for f in listdir(Folder) if isfile(join(Folder, f))]
    
    Li = []
    for file in onlyfiles:
        if file.split('.')[-1] == 'csv':
            Li.append(pd.read_csv(Folder + '/' + file, index_col=None, header=0))
            
    frame = pd.concat(Li, axis=0, ignore_index=True)
    
    frame['ds'] = pd.to_datetime(frame['Last Update'])
    return frame

def Read_APIData():
    r = requests.get(url = 'https://api.covid.bio/')
    return r.json()