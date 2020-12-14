# Report Project Mines ML 2020

## Introduction
This work is part of a project of the Machine Learning course presented by Mr. Christophe Cerisara at Ecole des Mines de Nancy [[1]](#1). 
The goal is to build a Deep Learning model to predict the traffic volume in Austin city, based on a Radar Traffic Data on Kaggle [[2]](#2).

## Data description
Our Traffic Data was collected from several radar sensors in the City of Austin and was augmented with geo-coordinates of the locations.
We have 4603861 and 12 columns. The columns are the following:
* **location_name**: Radar traffic sensor location name
* **location_latitude, location_longitude**: geo coordinates of the traffic sensor
* **Year, Month, Day, Day of Week, Hour, Minute**: Numeric time values for the observation
* **Time Bin** : Shows hour and minute
* **Direction** : Direction of traffic (SB:South bound, NB:North bound, EB: East bound, WB: West bound or None)
* **Volume**: Number of vehicles detected by the sensor, this is our target variable.

#### Remark:
We have data over the years: 2017, 2018, 2019, days of week vary from 0 to 6, and hours from 0 to 23.

## Objective
Our objective is to predict the traffic volume per hour for each day in January 2020. To do that we will build a model than given a location and direction predicts this traffic volume. The couple (location, direction) should be among the couples in our dataset. And the idea is to transform the data into a Time Series for each (location, direction). In the next section we will explain in detail the data preprocessing.

## Data preprocessing
In this section we will present the preprocessing phase. First, we are going to drop the following columns for the reasons below: <br>
* Time bin: because this information is already given by the Hour and Minute columns
* location_latitude, location_longitude: we will only need location_name for the prediction
* Day of week: we will not use this information, because we want to build our time series as a series with a 1 hour unit for each day like: 2017-09-03 02:00:00, 2017-09-03 03:00:00...  <br>
Next for a given day, location and direction, we will sum the traffic volume over minutes to obtain the traffic per hour.
The figure below shows an extract of the data we get after these transformations.<br>
<img src="https://user-images.githubusercontent.com/44069155/101657104-c955df00-3a43-11eb-97e8-a6adda17d239.png" width="50%"/>
<br>
So for each couple (location, direction) we will have a Time Series of traffic volume per hour, for which we defined a function *Get_Time_Series*.
Next we want to make our Time Series ready for a supervised Machine Learning algorithm, and create multiple samples for the training phase of our model. We will first use the growing window technique, then we will use the sliding window technique which resulted in a better performance in our case. For the model we will use Convolutionnal Neural Networks who proved to be very efficient in Time Series forecasting. We will do numerous experimentations for the prediction by trying different time lags (prediction for a week, a month..) which can be useful in case of objectives that are different than the one we chose. <br>

#### Remark:
* Concerning the feature scaling, we will normalize our data within the function *split_ts* defined later. We choose the Standardization method, by substracting the mean of the sequence and dividing by the standard deviation.
* We will see that we will predict the traffic for 1 month based on 3 months data with contains 24 hours for 30 days in 3 months with means 2160 entries. Therefore we need to make sure that we have enough

## Model: Convolutional Neural Network
As we explained previously, we will use convolutionnal neural networks to predict the traffic volume, in this section we will explain in detail the structure of the model and the data. Our approach was inspired by Mr. Christophe Cerisara's code for a project of sales prediction using Convolutionnal Neural Networks [[3]](#3).

### The sliding window
We first need to make our data ready for the neural network by creating multiple samples with their labels. Here we will present our approach for a particular example which is the prediction of the traffic volume for January 2020 for each day and for each hour, which gives as an output of length 24\*30 (24 hours and 30 days), the prediction will be based on the previous 3 months, so the window size will be 24\*30\*3. Our sliding window will move by time-step equal to the length of the output. The function *split_ts*  takes as inputs the location and direction and returns a list of samples and a list of their labels. Then we defined a function *train_test_set* that creates the training set and test set from the output of the function *split_ts*, the test set size is 20\% of the data. The data is not randomly shuffled so that the windows are built on consecutive samples, and to ensure that the test results are realistic, being evaluated on data collected after training the model [[4]](#4).

### Structure of the Neural network
In this section we will present the structure of our network. We used a 1D convolutional Neural Network with 2 convolutionnal layers both with kernel size equal to 3 and two fully connected layers. We used an Adam optimizer with a learning rate equal to 0.001 and an MSE loss. The *in_channels* number is equal to 1 and the output size is 24\*30 is the case of prediction for January 2020. We tried different values for the number of neurons in the hidden layers, we added a dropout layer to prevent the model from overfitting.

### General Approach
After we defined the neural network and preprocessed our data. We created a dictionnary whose keys are the couples (location,direction) and whose values are the time series for the corresponding couple. So for each couple we trained and evaluated our model on the corresponding data. The training loss decreases when the epochs increase, however we had some trouble in making the test_loss decrease, which suggested an overfitting problem. Therefore we added the dropout layer and tried to decrease the number of neurons in our neural network which gave better results.

## Results
In this section we will present the results we obtained in terms of model performance and prediction. The figures below show some of the losses we got for some couples of location name and direction.
#### Results for some locations and their directions
**For location=' CAPITAL OF TEXAS HWY / CEDAR ST' and direction='NB'**, we get 12 samples of size 2160 (hours in 3 months) each with a label of size 720 (hours in 1 month). After the training we get a training loss in the range of 0.01 and a test loss of the range 0.05. The figures below show the results for 500 epochs, we can see that the test loss doesn't quite improve after the epoch 50 but the training loss does decearse.
<br>
<img src="https://user-images.githubusercontent.com/44069155/101657104-c955df00-3a43-11eb-97e8-a6adda17d239.png" width="50%"/>
<br>
<br>
<img src="https://user-images.githubusercontent.com/44069155/101657104-c955df00-3a43-11eb-97e8-a6adda17d239.png" width="50%"/>
<br>
**For location=' LAMAR BLVD / ZENNIA ST' and direction='NB'**, we get 13 samples with size 2160. The figures below show the values of the training loss and test loss after each 50 epochs. We got a final training loss in the range of 0.03 and the test loss was in the range 0.07.
<br>
<img src="https://user-images.githubusercontent.com/44069155/101657104-c955df00-3a43-11eb-97e8-a6adda17d239.png" width="50%"/>
<br>
<br>
<img src="https://user-images.githubusercontent.com/44069155/101657104-c955df00-3a43-11eb-97e8-a6adda17d239.png" width="50%"/>
<br>

## References 
<a id="1">[1]</a> 
https://members.loria.fr/CCerisara/#courses/machine_learning/ <br>
<a id="2">[2]</a> 
https://www.kaggle.com/vinayshanbhag/radar-traffic-data<br>
<a id="3">[3]</a> 
http://talc2.loria.fr/xtof/ml/sales.py<br>
<a id="4">[4]</a> 
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb<br>
