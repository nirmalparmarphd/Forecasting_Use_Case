from zenml import pipeline
from .steps import *

# data processing pipeline
@pipeline(enable_cache=False, name='Data Loading and Preprocessing')
def data_pipeline():
    df = load_csv_data(path='training_data.csv')
    df = date_col_transform(df)
    df = date_col_extract(df)
    df_agg = agg_month_data(df)
    df_agg = date_add_agg_level(df_agg)
    df_agg = time_series_selection(df_agg)
    df_out = data_selection(df_agg)
    return df_out

@pipeline(enable_cache=False, name='AutoARIMA Forecasting')
def training_pipeline_AutoARIMA(data):
    model_ARIMA = forecasting_model('AutoARIMA')
    mape = forecasting_model_training_ARIMA(model_ARIMA, data)
    return mape

@pipeline(enable_cache=False, name='SeasonalNaive Forecasting')
def training_pipeline_SeasonalNaive(data):
    model_SeasonalNaive = forecasting_model('SeasonalNaive')
    mape = forecasting_model_training_SeasonalNaive(model_SeasonalNaive, data)
    return mape

@pipeline(enable_cache=False, name='Forecasting Pipeline')
def main_pipeline():
    data = data_pipeline()
    mape_arima = training_pipeline_AutoARIMA(data)
    mape_sn = training_pipeline_SeasonalNaive(data)
    