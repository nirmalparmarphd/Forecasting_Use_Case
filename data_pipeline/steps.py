from zenml import step
from typing import Any, Annotated, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error
from zenml.materializers.base_materializer import BaseMaterializer
from statsforecast.core import StatsForecast
from statsforecast.models import (
                                    HoltWinters,
                                    CrostonClassic as Croston, 
                                    HistoricAverage,
                                    DynamicOptimizedTheta as DOT,
                                    SeasonalNaive,
                                    AutoARIMA,
                                    SeasonalExponentialSmoothing,
                                    AutoETS
                                )

statsforecast_base = StatsForecast
from zenml.enums import ArtifactType
model_AutoARIMA = AutoARIMA()

# class StatsForecast_Custom(BaseMaterializer):
#     ASSOCIATED_TYPES = (AutoARIMA())
#     ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL




#1
@step
def load_csv_data(path:str) -> Annotated[pd.DataFrame, 'CSV Data']:
    """
    loading csv file data
    """
    df = pd.read_csv(path)
    return df

#2
@step
def date_col_transform(df:pd.DataFrame, col:str='DATUM') -> Annotated[pd.DataFrame, 'Timestamp formatting']:
    """
    to convert DATUM to timestamp format
    """
    df[col] = pd.to_datetime(df[col])
    return df

#3
@step
def date_col_extract(df:pd.DataFrame) -> Annotated[pd.DataFrame, 'Extracting date and months']:
    """
    extracting month and year value from DATUM columns and saving as new columns.
    """
    col='DATUM'
    df['DATUM_MONTH'] = df[col].dt.month
    df['DATUM_YEAR'] = df[col].dt.year 
    return df

#4
@step
def agg_month_data(df:pd.DataFrame) -> Annotated[pd.DataFrame, 'Monthly aggregation']:
    """
    aggregating data on monthly level for each HKONT.
    """
    df_agg = df.groupby([df['HKONT'], df['DATUM_YEAR'], df['DATUM_MONTH']])['VALUE_EUR'].sum().reset_index()
    return df_agg

#5
@step
def date_add_agg_level(df:pd.DataFrame) -> Annotated[pd.DataFrame, 'Monthly aggregation - Date']:
    """
    adding a column for date after aggregating at month level
    setting date as index
    """
    df['DATUM_AGG'] = pd.to_datetime(df['DATUM_YEAR'].astype(str) + '-' + 
                                     df['DATUM_MONTH'].astype(str) + '-1')
    # df.set_index('DATUM_AGG', inplace=True)
    return df

#6
@step
def data_selection(df:pd.DataFrame) -> Annotated[pd.DataFrame, 'Date Processing']:
    """
    formatting columns as per statsforecast module
    """
    col = ['DATUM_AGG', 'HKONT', 'VALUE_EUR']
    col_sf = ['ds', 'unique_id', 'y']
    data_ = df[col]
    data_.columns = col_sf
    return data_

#7
@step
def time_series_selection(df:pd.DataFrame) -> Annotated[pd.DataFrame, 'Time-Series selection']:
    """
    for selecting time-series with appropriate data points (hkont: timestamps)
    """
    ts_dict = (df.HKONT.value_counts(ascending=False)).to_dict()
    ts_dict = {key: value for key, value in ts_dict.items() if value < 12}
    for i in ts_dict.keys():
        data_idx = df[df['HKONT'] == i].index
        df = df.drop(data_idx)
    return df

#8
@step
def forecasting_model(model_name:str)->Annotated[Union[AutoARIMA, SeasonalNaive], 'Statsforecast Model']:
    if model_name.lower() == 'autoarima':
        model = AutoARIMA(season_length=6, seasonal=True)
    elif model_name.lower() == 'seasonalnaive':
        model = SeasonalNaive(season_length=6)
    else:
        raise ValueError
    return model

#9
@step
def forecasting_model_training_ARIMA(model:AutoARIMA, data:pd.DataFrame)->Annotated[np.float32, 'trained_model_AutoARIMA']:
    
    # model = AutoARIMA(season_length=6, seasonal=True)

    #solver
    sf = StatsForecast(df=data, 
                    models=[model], 
                    freq="M", 
                    n_jobs=-1)
    
    #cv
    crossvalidation_df = sf.cross_validation(df = data,
                                            h = 6,
                                            step_size = 1,
                                            n_windows = 3)
    crossvalidation_df.fillna(0, inplace=True)
    
    #model fit
    sf.fit()

    # forecast
    forecast_df = sf.forecast(df=data, h=6, level=[90])
    forecast_df.fillna(0, inplace=True)

    # Evaluate model
    mape = mean_absolute_percentage_error(crossvalidation_df['y'], crossvalidation_df['AutoARIMA'])
    return mape

#9
@step
def forecasting_model_training_SeasonalNaive(model:SeasonalNaive, data:pd.DataFrame)->Annotated[np.float32, 'trained_model_SeasonalNaive']:
    
    # model = AutoARIMA(season_length=6, seasonal=True)

    #solver
    sf = StatsForecast(df=data, 
                    models=[model], 
                    freq="M", 
                    n_jobs=-1)
    
    #cv
    crossvalidation_df = sf.cross_validation(df = data,
                                            h = 6,
                                            step_size = 1,
                                            n_windows = 3)
    crossvalidation_df.fillna(0, inplace=True)
    
    #model fit
    sf.fit()

    # forecast
    forecast_df = sf.forecast(df=data, h=6, level=[90])
    forecast_df.fillna(0, inplace=True)

    # Evaluate model
    mape = mean_absolute_percentage_error(crossvalidation_df['y'], crossvalidation_df['SeasonalNaive'])
    return mape

@step
def best_model(model_dict:dict)->Annotated[dict, 'Best Model: MAPE']:
    min_rmse = min(model_dict.values())
    min_rmse_models = {model: rmse for model, rmse in model_dict.items() if rmse == min_rmse}
    return min_rmse_models



    