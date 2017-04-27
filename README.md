# AirQualityImpute_ML
Impute missing air quality data from the [EPA AQS](https://www.epa.gov/outdoor-air-quality-data) dataset with models trained on surrounding air quality stations. Then, predict the air quality at a point using a method similar to that of [Falke and Husar](http://capita.wustl.edu/capita/capitareports/mappingairquality/mappingaqi.pdf).

The code is currently setup to predict the PM10 concentration at a point. Data missing from PM10 monitors is filled in with models trained on surrounding PM10, PM2.5, CO, and/or ozone data.

## Filling in missing air quality readings
Many EPA AQS sites only report air quality values once every 3, 6, or 12 days, and stations frequently experience downtime. To fill in missing data and create a more complete datase, neural networks are trained to predict stations' readings based on available readings from stations nearby.

The figure below shows the available (black) and filled-in (red) data for a PM10 station in Florida. The filled-in data is the output of a model trained on the available data, with surrounding air quality stations (whose readings are shown below the plot) as predictors. For this station, data is only reported through 2012; values for remaining days are filled in with the model.

![Imputation of missing PM10 data](/documentation/example_imputation.PNG)

## Estimating the air quality at a point
Values reported or imputed for the stations surrounding a point are used to predict the air quality at the point, based on the weighted averaging of [Falke and Husar](http://capita.wustl.edu/capita/capitareports/mappingairquality/mappingaqi.pdf). This approach scales a station's weight with the inverse square of the distance and accounts for the spatial clustering of stations. The temporal considerations presented in the paper are not included in the weighting here.

## Validation
To test the algorithm, one EPA PM10 station is chosen as a "test" station and its readings are removed from the database and reserved for validation.

The map below shows the point at which the PM10 concentration is being estimated (the red X), the EPA AQS PM10 stations used in this estimation (the black circles), and the other air quality stations used to impute missing values in the PM10 stations (the colored triangles: black are other PM10 stations, green are PM2.5 stations; magenta are CO stations).

![EPA AQS stations used in spatial interpolation and modelling](/documentation/example_map.PNG)

Below is a plot of the target data recorded at the red X which was reserved for validation (green), and the estimation as determined with (black) and without (grey) missing PM10 data filled in. Both daily values (top) and a 14-day rolling mean (bottom) are shown. The rolling value is relevant to correlating PM10 with PV soiling trends, since we define soiling periods to be at least 14 days in length.

![Comparison to target data, with and without missing data filled in](/documentation/example_validation.PNG)

## Use
To run this validation, download air quality data (at least PM10 and PM2.5) [from the EPA](https://aqsdr1.epa.gov/aqsweb/aqstmp/airdata/download_files.html#Daily) and put the location of the directory with the `.csv`s in `folder` in `AQ_ML.py`. Run the script `aq_pipeline.py`, in which the coordinates of a PM10 station for which there is daily reported data are stored in `latlon`. Data from this station is removed from the dataset and used in validation (as shown above). Choose `r_max_interp` and `r_max_ML` such that a suitable number of stations are chosen for interpolation and creating the models; this step may be automated in the future. 
