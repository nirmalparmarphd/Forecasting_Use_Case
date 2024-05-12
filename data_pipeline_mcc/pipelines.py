from zenml import pipeline
from .steps import *
from zenml.logger import get_logger

logger = get_logger(__name__)

# data processing pipeline
@pipeline(enable_cache=True, name='Data Loading and Preprocessing')
def data_pipeline():
    df = load_csv_data(path='training_data.csv')
    df = date_col_transform(df, col='DATUM')
    df = date_col_transform(df, col='DUE_DATE')
    df = date_col_transform(df, col='PAYMENT_DATE')
    df = inconsist_col_drop(df)
    df = label_calculation(df)
    return df

# feature engineering
@pipeline(enable_cache=True, name='Feature Engineering')
def feature_engineering_pipeline(data):
    mcc_info = multi_class_count_check(data)
    data = feature_selection(data)
    x_train, x_test, y_train, y_test = data_splitter(data)
    return x_train, x_test, y_train, y_test
    
# ML training
@pipeline(enable_cache=True, name='ML Training')
def model_training_pipeline(x_train, x_test, y_train, y_test, model_name:str='rfc'):    
    model_mcc = model_selection(model_name)
    model_mcc = model_training(model_mcc, x_train, y_train)
    predicitons = model_prediction(model_mcc, x_test)
    confu_matrix, mcc_report = model_evaluation(predicitons, y_test)
    return confu_matrix, mcc_report 

# main pipeline
@pipeline(enable_cache=True, name='Main MCC Pipeline')
def main_mcc_pipeline():
    data = data_pipeline()
    x_train, x_test, y_train, y_test = feature_engineering_pipeline(data)
    confu_matrix_rfc, mcc_report_rfc = model_training_pipeline(x_train, x_test, y_train, y_test, model_name='rfc')
    confu_matrix_rfc, mcc_report_rfc = model_training_pipeline(x_train, x_test, y_train, y_test, model_name='abc')
