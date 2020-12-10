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
from random import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
"""
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
"""

###########################################
##DATA PREPROCESSING
###########################################


data = pd.read_csv('C:\\Users\\33758\\Desktop\\3A Mines\\Machine learning\\Radar Traffic\\Radar_Traffic_Counts.csv')

##The idea behind the following transformations is to have a Time Series 
##of Volume as a function of the date for each couple (location_name,direction)
##The date unit will be 1 hour

##dropping some columns
#we drop the columns we will not need in our model
#we drop time bin, this information is already in Hour and Minute
data.drop('Time Bin', inplace=True, axis=1)
#we will base our model only on location name
data.drop('location_latitude', inplace=True, axis=1)
data.drop('location_longitude', inplace=True, axis=1)
#Our time series will vary by date: hour-day/month/year
#we drop day of week column
data.drop('Day of Week', inplace=True, axis=1)
data.columns

##grouping year month day and hour into one datatime column "Date"
#each Date value is the hour of a given day (day/month/year)
date1 = data[['Year','Month','Day','Hour']] #save year, month day and hour to put them later in one column
data.drop('Month', inplace=True, axis=1)
data.drop('Day', inplace=True, axis=1)
data.drop('Hour', inplace=True, axis=1)
data[['Year']]=pd.to_datetime(date1,unit='D')
data=data.rename(columns={"Year": "Date"})#rename the column of date

#sort values-grouping by location name, date, minute for a better vizualization
data.sort_values(['location_name','Date','Minute'],inplace=True)

#we sum the volume over minutes for each hour-Date-
group= data.groupby(by = ['location_name','Date','Direction'], as_index=False)['Volume'].sum()
group.head()

#rearrange data
d = {'location_name': group['location_name'], 'Direction': group['Direction'],'Date': group['Date'],'Volume':group['Volume']}
data2=pd.DataFrame(data=d)
##This is the data we will use from now on 

###########################################
##count number of couples (location, direction)
###########################################

count_u = data2.groupby(['location_name','Direction']).size().reset_index().rename(columns={0:'count'})
#40 rows=40 couples (location, direction)
count_u.info()
#let's check if all the couples have sufficient data
min(count_u['count']) #1
max(count_u['count']) # 17206
count_u['count'].mean() #12392.15
#we delete couples who have less than 100 counts
new_data = count_u[count_u['count'] > 100] #6 couples were deleted, 34 are left
#for each location and direction we will have a time series
#where the volume is a function of "Date-Hour"


def Get_Time_Series(location,direction):
    #location and direction are strings
    extract=data2.loc[data2.location_name==location][data2.Direction==direction]
    #dates=extract['Date']
    volume=extract['Volume'].to_numpy()
    return volume
#test
#loc=' BURNET RD / PALM WAY (IBM DRIVEWAY)'
#direct='NB'
#Get_Time_Series(loc,direct)
    
##list of location_names
names=data2['location_name'].unique().tolist()
#list of directions 
directions=['NB','SB','EB','WB','None']


###########################################
## the teacher's code
###########################################

##convolutional neural network
class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(8)
        )
        self.fc1 = nn.Linear(in_features=64*8, out_features=24*7*2)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=24*7*2, out_features=24*7)
        self.fc3 = nn.Linear(in_features=24*7, out_features=24*7)
 
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
xmin, xmax = 100.0, -100.0
vnorm = 1000.0
#we want to predict traffic per hour for a day  based on the traffic of 
#the previous 3 months (per hour each day)
minlen = 24*30*2
# if <8 then layer1 outputs L=7/2=3 which fails because layer2 needs L>=4
#testing on an example
si2X, si2Y = {}, {}
dsi2X, dsi2Y = {}, {}
#for s,i in dsi2c.keys():
#    seq = dsi2c[(s,i)]
seq=Get_Time_Series(names[0], directions[0])[10000:]

xlist, ylist = [], []
for m in range(minlen, len(seq)-24*7*2):
    #the growing window has initial size minlen
    #m moves  by 1 hour
    if m==minlen:
        xx = [seq[z]/vnorm for z in range(m)]
    else:
        m=m+24*7-1
        xx = [seq[z]/vnorm for z in range(m)] #add the last day to our growing window
    if max(xx)>xmax: xmax=max(xx)
    if min(xx)<xmin: xmin=min(xx)
    xlist.append(torch.tensor(xx,dtype=torch.float32))
    yy = [seq[m+k]/vnorm for k in range(24*7)]
    ylist.append(torch.tensor(yy,dtype=torch.float32))
si2X= xlist
si2Y= ylist
if True: # build evaluation dataset
    xx = [seq[z]/vnorm for z in range(len(seq)-24*7-1)]
    dsi2X = [torch.tensor(xx,dtype=torch.float32)]
    yy = [seq[len(seq)-1-24*7+i]/vnorm for i in range(24*7)]
    dsi2Y= [torch.tensor(yy,dtype=torch.float32)]
"""
minlen = 24*30*2
# if <8 then layer1 outputs L=7/2=3 which fails because layer2 needs L>=4
#testing on an example
si2X, si2Y = {}, {}
dsi2X, dsi2Y = {}, {}
#for s,i in dsi2c.keys():
#    seq = dsi2c[(s,i)]
seq=Get_Time_Series(names[0], directions[0])[10000:]

xlist, ylist = [], []
for m in range(minlen, len(seq)-24*2):
    #the growing window has initial size minlen
    #m moves  by 1 hour
    if m==minlen:
        xx = [seq[z]/vnorm for z in range(m)]
    else:
        m=m+23
        xx = [seq[z]/vnorm for z in range(m)] #add the last day to our growing window
    if max(xx)>xmax: xmax=max(xx)
    if min(xx)<xmin: xmin=min(xx)
    xlist.append(torch.tensor(xx,dtype=torch.float32))
    yy = [seq[m+k]/vnorm for k in range(24)]
    ylist.append(torch.tensor(yy,dtype=torch.float32))
si2X= xlist
si2Y= ylist
if True: # build evaluation dataset
    xx = [seq[z]/vnorm for z in range(len(seq)-24-1)]
    dsi2X = [torch.tensor(xx,dtype=torch.float32)]
    yy = [seq[len(seq)-1-24+i]/vnorm for i in range(24)]
    dsi2Y= [torch.tensor(yy,dtype=torch.float32)]
"""
print("ntrain %d %f %f" % (len(si2X),xmin,xmax))
#len(xlist[0]) 720 (24*30 heures dans 1 mois)
#len(ylist[0])  24 (24 heures )
mod = TimeCNN()
loss = torch.nn.MSELoss()
opt = torch.optim.Adam(mod.parameters(),lr=0.01)

xlist = si2X
ylist = si2Y
idxtr = list(range(len(xlist)))
for ep in range(10):
    shuffle(idxtr)
    lotot=0.
    mod.train()
    for j in idxtr:
        opt.zero_grad()
        haty = mod(xlist[j].view(1,1,-1))
        # print("pred %f" % (haty.item()*vnorm))
        lo = loss(haty,ylist[j].view(1,-1))
        lotot += lo.item()
        lo.backward()
        opt.step()

    # the MSE here is computed on a single sample: so it's highly variable !
    # to make sense of it, you should average it over at least 1000 (s,i) points
    mod.eval()
    haty = mod(dsi2X[0].view(1,1,-1))
    lo = loss(haty,dsi2Y[0].view(1,-1))
    print("epoch %d loss %1.9f testMSE %1.9f" % (ep, lotot, lo.item()))


## Sliding window
minlen = 24*30*2
# if <8 then layer1 outputs L=7/2=3 which fails because layer2 needs L>=4
#testing on an example
si2X, si2Y = {}, {}
dsi2X, dsi2Y = {}, {}
#for s,i in dsi2c.keys():
#    seq = dsi2c[(s,i)]
seq=Get_Time_Series(names[0], directions[0])

#building the sliding window
n_steps=24*30*2 #2 months
horizon=24*7 #1 week
xlist, ylist = [], []
for i in range(len(seq)//horizon):
    end= i*horizon + n_steps
    if end > len(seq)-1:
        break
    xx = seq[i*horizon:end]/vnorm
    xlist.append(torch.tensor(xx,dtype=torch.float32))
    yy = seq[end:(end+horizon)]/vnorm
    ylist.append(torch.tensor(yy,dtype=torch.float32))
si2X= xlist
si2Y= ylist
if True: # build evaluation dataset
    xx = seq[:len(seq)-24*7]/vnorm
    dsi2X = [torch.tensor(xx,dtype=torch.float32)]
    yy = seq[len(seq)-24*7:]/vnorm 
    dsi2Y= [torch.tensor(yy,dtype=torch.float32)]

print("ntrain %d %f %f" % (len(si2X),xmin,xmax))
#len(xlist[0]) 720 (24*30 heures dans 1 mois)
#len(ylist[0])  24 (24 heures)
mod = TimeCNN()
loss = torch.nn.MSELoss()
opt = torch.optim.Adam(mod.parameters(),lr=0.01)

xlist = si2X
ylist = si2Y
idxtr = list(range(len(xlist)))
for ep in range(10):
    shuffle(idxtr)
    lotot=0.
    mod.train()
    for j in idxtr:
        opt.zero_grad()
        haty = mod(xlist[j].view(1,1,-1))
        # print("pred %f" % (haty.item()*vnorm))
        lo = loss(haty,ylist[j].view(1,-1))
        lotot += lo.item()
        lo.backward()
        opt.step()

    # the MSE here is computed on a single sample: so it's highly variable !
    # to make sense of it, you should average it over at least 1000 (s,i) points
    mod.eval()
    haty = mod(dsi2X[0].view(1,1,-1))
    lo = loss(haty,dsi2Y[0].view(1,-1))
    print("epoch %d loss %1.9f testMSE %1.9f" % (ep, lotot, lo.item()))
