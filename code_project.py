# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 12:12:09 2020

@author: Loubna Ben Allal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from random import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision

""" We are intrested in traffic volume prediction in Austin City""""

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
##list of location_names
names=data2['location_name'].unique().tolist()
#list of directions 
directions=['NB','SB','EB','WB','None']
    
        
date = data2.groupby(['location_name','Direction']).size()
count_u = data2.groupby(['location_name','Direction']).size().reset_index().rename(columns={0:'count'})
#40 rows=40 couples (location, direction)
count_u.info()
#let's check if all the couples have sufficient data
min(count_u['count']) #1
max(count_u['count']) # 17206
count_u['count'].mean() #12392.15
count_u=count_u.sort_values(['count'])
#we delete couples who have less than 100 counts
new_data = count_u[count_u['count'] > 4000] #7 couples were deleted, 34 are left
min(new_data['count']) #11491
#for each location and direction we will have a time series
#where the volume is a function of "Date-Hour"


def Get_Time_Series(location,direction,dates=False):
    #location and direction are strings
    extract=data2.loc[data2.location_name==location][data2.Direction==direction]
    volume=extract['Volume']
    if dates==False:
        return volume.to_numpy()
    else:
        #we will need this part later for the plots
        dates=extract['Date']
        frame = {'date': dates,'volume': volume} 
        dataf=pd.DataFrame(frame)
        return dataf
#test
#seq=Get_Time_Series(names[0], directions[0])

###################################################################
##DICTIONNARY of the locations and directions as key (couple) and the TRAFFIC VOLUME as a value
###################################################################   
data_dict={}
couples=new_data[['location_name','Direction']]
couples=[tuple(couples.iloc[i]) for i in range(couples.shape[0])]
volume=[]
for couple in couples:
    location,direction=couple
    volume.append(Get_Time_Series(location,direction))
for i in range(len(volume)):
    key=couples[i] #(location,direction)
    data_dict[key]=volume[i]  
    
#the data will be normalized later
    
###########################################
## SLIDING WINDOW
###########################################
#building the sliding window
#n_steps=24*30*2 #2 months
#horizon=24*7 #1 week
#min count in data is 11491 we will have a lot more than 7(mincount//n_steps) samples-we only move the window by 24*7

def split_ts(seq,horizon=24*30,n_steps=24*30*3):
    """ this function take in arguments a traffic Time Series for the couple (l,d)
    and applies a sliding window of length n_steps to generates samples having this 
    length and their labels (to be predicted) whose size is horizon
    """
    #for the Min-Max normalization X-min(seq)/max(seq)-min(seq)
    max_seq=max(seq)
    min_seq=min(seq)
    seq_norm=max_seq-min_seq
    xlist, ylist = [], []
    for i in range(len(seq)//horizon):
        end= i*horizon + n_steps
        if end+horizon > len(seq)-1:
            break
        xx = (seq[i*horizon:end]-min_seq)/seq_norm
        xlist.append(torch.tensor(xx,dtype=torch.float32))
        yy = (seq[end:(end+horizon)]-min_seq)/seq_norm
        ylist.append(torch.tensor(yy,dtype=torch.float32))
    print("number of samples %d and sample size %d (%d months)" %(len(xlist),len(xlist[0]),n_steps/(24*30)))
    return(xlist,ylist)

###########################################
## Model: CNN
###########################################
#goal: Predict one MONTH traffic(for each day per hour) based on the past 3/5 months
##convolutional neural network with a sliding window
#this network structure and the splitting method were inspired from Mr Christophe Cerisara's code for a problem of sales prediction
class TimeCNN(nn.Module):
    def __init__(self):
        super(TimeCNN, self).__init__()
        #1st convolutional layer
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        #2nd convolutional layer
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(24*14)
        )
        #fully connected layers
        self.fc1 = nn.Linear(in_features=64*24*14, out_features=120)
        self.drop = nn.Dropout2d(0.3)
        self.fc2 = nn.Linear(in_features=120, out_features=24*30)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out


def train_test_set(xlist,ylist):
    """ this functions splits the samples and labels datasets xlist and ylist
    (given by the function split_ts) into a training set and a test set
    """
    data_size=len(xlist)
    test_size=int(data_size*0.2) #20% of the dataset
    #training set
    X_train  = xlist[:data_size-test_size]
    Y_train = ylist[:data_size-test_size]
    #test set
    X_test = xlist[data_size-test_size:]
    Y_test = ylist[data_size-test_size:]
    return(X_train,Y_train,X_test,Y_test)

def model_traffic(mod,seq,num_ep=60,horizon=24*30,n_steps=24*30*3):
    #inputs are the model mod, the Time Series sequence and the number of epochs
    #building the model
    xlist,ylist = split_ts(seq,horizon,n_steps)
    X_train,Y_train,X_test,Y_test=train_test_set(xlist,ylist)
    idxtr = list(range(len(X_train)))
    #loss and optimizer
    loss = torch.nn.MSELoss()
    opt = torch.optim.Adam(mod.parameters(),lr=0.0005)
    loss_val_train=[]
    loss_val_test=[]
    for ep in range(num_ep):
        shuffle(idxtr)
        ep_loss=0
        test_loss=0
        mod.train()
        for j in idxtr:
            opt.zero_grad()
            #forward pass
            haty = mod(X_train[j].view(1,1,-1))
            # print("pred %f" % (haty.item()*vnorm))
            lo = loss(haty,Y_train[j].view(1,-1))
            #backward pass
            lo.backward()
            #optimization
            opt.step()
            ep_loss += lo.item()
        loss_val_train.append(ep_loss)
        #model evaluation
        mod.eval()
        for i in range(len(X_test)):    
            haty = mod(X_test[i].view(1,1,-1))
            test_loss+= loss(haty,Y_test[i].view(1,-1)).item()
        loss_val_test.append(test_loss)
        if ep%50==0:
            print("epoch %d training loss %1.9f test loss %1.9f" % (ep, ep_loss, test_loss))
    #test_loss is given for the selected model (last epoch)
    epochs=[i for i in range(num_ep)]
    fig, ax = plt.subplots()
    ax.plot(epochs,loss_val_train,label='training loss')
    ax.plot(epochs,loss_val_test,label='test loss')
    ax.legend()
    plt.show()
    return ep_loss,test_loss

    
###################################################################
# TRAINING AND EVALUATION OF THE MODEL FOR EACH (LOCATION,DIRECTION)
###################################################################

results =pd.DataFrame( columns = ["couple", "training_loss", "test_loss"])
num_ep=500
horizon=24*30
n_steps=24*30*5
for l,d in list(data_dict.keys())[5:]:
    couple=(l,d)
    seq=data_dict[(l,d)] #volume sequence for (l,d) location, direction
    xlist,ylist = split_ts(seq,horizon,n_steps)
    print("couple:",(l,d))
    print("number of samples in the dataset:", len(xlist))
    mod = TimeCNN()
    train_loss, test_loss =model_traffic(mod,seq,num_ep,horizon,n_steps)
    print("train_loss, test_loss =", train_loss, test_loss, "\n")
    results.loc[len(results)] = [couple, train_loss, test_loss]
    del(mod)

###################################################################
# TRAFFIC PREDICTION
###################################################################

"""prediction of the traffic for December 2019 based on the previous 5 months
#=last sample in X_test
#plot the predictions Vs real values for test set
#for the first couple in the dictionnary
l,d=list(data_dict.keys())[0]
#Get_Time_Series(location,direction,dates=False)
seq=data_dict[(l,d)] #volume sequence for (l,d) location, direction
xlist,ylist = split_ts(seq,horizon,n_steps)
print("couple:",(l,d))
print("number of samples in the dataset:", len(xlist))
mod = TimeCNN()
train_loss, test_loss =model_traffic(mod,seq,350,horizon,n_steps)
X_train,Y_train,X_test,Y_test=train_test_set(xlist,ylist)

true_seq=list(X_test[len(X_test)-2].detach().numpy())+list(Y_test[len(Y_test)-2].detach().numpy())
pred_seq=mod(X_test[len(X_test)-2].view(1,1,-1)).detach().numpy()
pred_seq=list(np.transpose(pred_seq).reshape(720))
index=[ i for i in range(len(true_seq))]
index_y=[i for i in range(len(true_seq)-len(pred_seq),len(true_seq))]
fig, ax = plt.subplots()
ax.set_title('Comparison between the predicted traffic and the real one')
ax.plot(index,true_seq,label='real traffic')
ax.plot(index_y,pred_seq,label='predicted traffic')
ax.legend()
plt.show()
"""

#Prediction for january 2020
l,d=list(data_dict.keys())[2]
seq=data_dict[(l,d)] #volume sequence for (l,d) location, direction
xlist,ylist = split_ts(seq,horizon,n_steps)
print("couple:",(l,d))
print("number of samples in the dataset:", len(xlist))
#del(mod)
#we train again because we didn't save the model 
mod = TimeCNN()
train_loss, test_loss =model_traffic(mod,seq,350,horizon,n_steps)
X_train,Y_train,X_test,Y_test=train_test_set(xlist,ylist)
#the last sequence of 3 months in 2019
last_seq=list(xlist[len(X_test)-1].detach().numpy())
fin=len(Y_test)
fin2=len(Y_test[fin-1])
ytest=list(Y_test[fin-1].detach().numpy())
for i in range(fin2):
    last_seq.append(ytest[i])
    
last_seq=torch.tensor(last_seq,dtype=torch.float32)
last_seq=last_seq[720:] #3600 len
pred_seq=mod(last_seq.view(1,1,-1)).detach().numpy()
pred_seq=list(np.transpose(pred_seq).reshape(720))
index=[ i for i in range(len(last_seq))]
index_y=[i for i in range(len(last_seq),len(last_seq)+len(pred_seq))]
fig, ax = plt.subplots()
ax.set_title('Prediction of January 2020')
ax.plot(index,last_seq,label='last 3 months of 2019')
ax.plot(index_y,pred_seq,label='traffic in January')
ax.legend()
plt.show()

