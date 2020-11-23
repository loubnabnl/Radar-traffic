# Report Project Mines ML 2020

## Introduction
This work is part of a project of the Machine Learning course presented by Mr. Christophe Cerisara at Ecole des Mines de Nancy [[1]](#1). 
The goal is to build a Deep Learning model to predict the traffic volume in Austin city, based on a Radar Traffic Data on Kaggle [[2]](#2).

## Data Description
Our Traffic Data was collected from several radar sensors in the City of Austin and was augmented with geo-coordinates of the locations.
We have 4603861 and 12 columns. The columns are the following:
* **location_name**: Radar traffic sensor location name
* **location_latitude, location_longitude**: geo coordinates of the traffic sensor
* **Year, Month, Day, Day of Week, Hour, Minute**: Numeric time values for the observation
* **Time Bin**
* **Direction** : Direction of traffic (SB:South bound, NB:North bound, EB: East bound, WB: West bound or None)
* **Volume**: Number of vehicles detected by the sensor, this is our target variable.
#### Remark:
We have data over the years: 2017, 2018, 2019, days of week vary from 0 to 6, and hours from 0 to 23.



## References
<a id="1">[1]</a> 
https://members.loria.fr/CCerisara/#courses/machine_learning/ <br>
<a id="2">[2]</a> 
https://www.kaggle.com/vinayshanbhag/radar-traffic-data
