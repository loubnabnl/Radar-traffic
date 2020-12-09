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
The figure below shows an extract of our data
<img src="dephell.png" width="50%"/> <br>
#### Remark:
We have data over the years: 2017, 2018, 2019, days of week vary from 0 to 6, and hours from 0 to 23.

## Objective
Our objective is to predict the traffic volume per hour for each day in January 2020. To do that we will build a model than given a location and direction predicts this traffic volume. The couple (location, direction) should be among the couples used for the training. And the idea is to transform the data into a Time Series for each (location, direction). In the next section we will explain in detail the data preprocessing.

## Data preprocessing
In this section we will present the preprocessing phase. First, we are going to drop the following columns for the reasons below: <br>
* Time bin: because this information is already given by the Hour and Minute columns
* location_latitude, location_longitude: we will only need location_name for the prediction
* Day of week: we will not use this information, because we want to build our time series as a series with a 1 hour unit for each day like: 2017-09-03 02:00:00, 2017-09-03 03:00:00...  <br>
Next for a given day, location and direction, we will sum the traffic volume over minutes to obtain the traffic per hour.
The figure below shows an extract of the data we get after these transformations
<img src="dephell.png" width="50%"/><br>
## References
<a id="1">[1]</a> 
https://members.loria.fr/CCerisara/#courses/machine_learning/ <br>
<a id="2">[2]</a> 
https://www.kaggle.com/vinayshanbhag/radar-traffic-data
