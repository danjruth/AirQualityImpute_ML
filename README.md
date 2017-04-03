# AirQualityImpute_ML
Impute missing air quality data from the [EPA AQS](https://www.epa.gov/outdoor-air-quality-data) dataset, then use the method of [Falke and Husar](http://capita.wustl.edu/capita/capitareports/mappingairquality/mappingaqi.pdf) to predict the air quality at a point.

## Usage
To estimate the air quality at a point while filling in values missing from the EPA AQS dataset, call `AQ_ML.py`'s function `predict_aq_vals(latlon,start_date,end_date,r_max_interp,r_max_ML,all_data)` where:

- `latlon` is the (latitude,longitude) of the location at which to predict the air quality
- `start_date` and `end_date` are the first and last days for which to predict the air quality
- `r_max_interp` is the radius (in kilometers) to look for monitor stations around `latlon` when implementing the prediction algorithm
- `r_max_ML` is the radius around each monitor stations to look for other monitor stations with which to create the model that'll be used to fill in any missing data
- `all_data` is a `pandas` `DataFrame` of all the data (within the dates of interest) contained in the .csv files provided by the EPA [here](https://aqsdr1.epa.gov/aqsweb/aqstmp/airdata/download_files.html#Daily)

The function `extract_raw_data` in `AQ_ML.py` can be called to create a dataframe containing readings taken between `start_date` and `end_date` from the raw .csv files.

To compare the predicted values (with and without missing values imputed) against data recorded by a monitor station at `latlon` (for validation), add `ignore_closest=True` when calling `predict_aq_vals`. Data from the station closest to `latlon` will be thrown out in predicting the data at the point and imputing stations' missing data. The function will return the predicted values obtained with and without missing values imputed as well as the known data that was ignored in the predictions.
