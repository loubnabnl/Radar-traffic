# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:12:09 2020

@author: 33758
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os


import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

data = pd.read_csv('C:\\Users\\33758\\Desktop\\3A Mines\\Machine learning\\Radar Traffic\\Radar_Traffic_Counts.csv')

#Visualization of the data
data.head()
data['location_name'].head()
data[[ 'location_latitude', 'location_longitude','Year','Month', 'Day',
      'Day of Week']].head(10)
data[['Hour', 'Minute', 'Time Bin','Direction', 'Volume']].head()
#shape
data.shape  #(4603861, 12)
#columns
data.columns
#info
data.info()

############Data Preprocessing############

#checking missing values
data.isnull().sum()
#No missing values in this dataset

#summary of the data
description=data.describe()
description.iloc[:,0:2]
description.iloc[:,2:5]
description.iloc[:,5:9]

#we can see that we have data over the years: 2017, 2018, 2019
#month 1:12
#day 1:31
#day of week 0:6
#Hour 0:23

#Visualization of the data
#data of the year 2017
data17=data[data['Year']==2017]
data17.shape[0]  #801 631
data17['Month'].unique()  #[12, 11, 10,  9,  8,  7,  6]
sorted(list(data17['Month'].unique())) #[6, 7, 8, 9, 10, 11, 12]
#for month 6
data17_M1=data17[data17['Month']==6]
data17_M1.shape[0] #3928
data17_M1['Day of Week'].unique() #[4, 3, 2, 1, 0]
#volume per day of week fo the year 2017
plt.plot(data17_M1['Day of Week'],data17_M1['Volume'])


#year 2018
data18=data[data['Year']==2018]
data18.shape[0]  #2 097 896
data18['Month'].unique() #[ 1,  8,  2,  9,  3, 10,  4,  5, 11, 12,  6,  7]
sorted(list(data18['Month'].unique())) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
#for month 1
data18_M1=data18[data18['Month']==1]
data18_M1.shape[0] #162188
data18_M1['Day of Week'].unique() #[2, 3, 4, 5, 6, 0, 1]
#volume per month fo the year 2017
plt.plot(data18_M1['Day of Week'],data18_M1['Volume'])

#year 2019
data19=data[data['Year']==2019]
data19.shape[0]  #1 704 334
data19['Month'].unique() #[ 9,  4,  1,  2,  3,  8,  7,  6,  5, 11, 10]
sorted(list(data19['Month'].unique())) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#for month 1
data19_M1=data19[data19['Month']==1]
data19_M1.shape[0] #135593
data19_M1['Day of Week'].unique() #[3, 4, 2, 5, 0, 1, 6]
#volume per month fo the year 2017
plt.plot(data19_M1['Day of Week'],data19_M1['Volume'])

#drpping time bin column
data.drop('Time Bin', inplace=True, axis=1)
data.columns

#grouping year month and day into datatime
date1 = data[['Year','Month','Day']]
data.drop('Month', inplace=True, axis=1)
data.drop('Day', inplace=True, axis=1)
data[['Year']]=pd.to_datetime(date1,unit='D')
data.rename(columns={"Year": "Date"})#doesn't work

#Drop day of week
data.drop('Day of Week', inplace=True, axis=1)

#sort values -grouping by location name, date.. for a better vizualization
data.sort_values(['location_name','Date','Hour','Minute'],inplace=True)

#liste=list(data.columns)
#liste[:8] #columns until Hour
#we group data by summing the volume of over minutes for the say hou(day, month..)
group= data.groupby(by = ['location_name', 'location_latitude','location_longitude','Year','Hour','Direction'])['Volume'].sum()
group=group.to_frame()
group.head()

#for each location and direction we will have a time series
#where the volume is a function of "Date-Hour"

##Standard scaler
#standard_X=preprocessing.StandardScaler()
data.head()
