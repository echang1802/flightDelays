# flightDelays

This code apply a random forest algorithm as a model to classified if a flight will be canceled or not. 

The variables given are:

Field            | Name Type | Description
-----------------|-----------|----------------------------------
Canceled         | Binary    | Canceled = 1
Month            | Integer   | Jan = 1
DepartureTime    | Integer   | Military Time (1:00 PM = 1300)
UniqueCarrier    | String    | Airline Carrier Code
SchedElapsedTime | Integer   | Scheduled Flight time in minutes
ArrDelay         | Integer   | Arrival delay in minutes
DepDelay         | Integer   | Departure delay in minutes
Distance         | Integer   | Distance in miles


## About modeling

### Feature enginering

* Two variables were created:
    1. DistanceGreater1000: If the distance is greater than 1000 miles or not. 
    2. DepartureAtLabourHour: If the flight is scheduled to departure between 6 and 21.

* The numeric variables were scaled.

* The dataset was balanced.

### Parameters settings.

A holdout set was use for tune the model hyper parameters, using a random search grid. 

## About code.

### Train.

To train a model use: 
~~~
python main.py
~~~

Adjust desire feature enginering on code. 

A benchmark model is setted using a logit model. 

### Predict

To make predictions over a new dataset use:
~~~
python predict.py filenameInput filenameOutput
~~~

Where:
* FilenameInput: Is the direction of the csv file with the dataframe.
* FilenameOutput: Is the direction where the dataframe with the predictions will be saved as a csv file. 
