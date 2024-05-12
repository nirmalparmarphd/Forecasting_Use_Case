from zenml import step
from typing import Any, Annotated, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, classification_report, confusion_matrix
from zenml.materializers.base_materializer import BaseMaterializer
from zenml.client import Client
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin
client = Client()


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
    if col == 'PAYMENT_DATE':
        df[col] = pd.to_datetime(df[col], format='%Y%m%d')
    else:
        df[col] = pd.to_datetime(df[col])
    return df

#3
@step
def inconsist_col_drop(df:pd.DataFrame)->Annotated[pd.DataFrame, 'Consistent Data']:
    df.drop(columns=['MANSP', 'CTLPC', 'HISTORICRATING', 'CURRENTRATING'], inplace=True)
    return df

#4
@step
def label_calculation(df:pd.DataFrame)->Annotated[pd.DataFrame, 'Label Created for MCC']:
    """
    0 = on time
    1 = before time # NOTE assumption made here
    2 = delay in time
    """
    df['DAY_COUNT'] = (df['DUE_DATE'] - df['PAYMENT_DATE']).dt.days
    
    def payment_check(value):
        if value >= 5:
            return 1
        elif value >= 0 and value < 5:
            return 0
        elif value < 0:
            return 2
        
    df['PAYMENT_STATUS'] = df['DAY_COUNT'].apply(payment_check)
    return df

#5
@step

def multi_class_count_check(df:pd.DataFrame)->Annotated[pd.Series, 'Value Counts MCC']:
    return df['PAYMENT_STATUS'].value_counts()

#6
@step
def feature_selection(df:pd.DataFrame)->Annotated[pd.DataFrame, 'Selected Features']:
    # Get columns with string data type
    string_columns = df.select_dtypes(include=['object', 'datetime64[ns]']).columns

    # Drop columns containing string values
    df = df.drop(columns=string_columns)
    df = df.drop(columns='DAY_COUNT')
    return df

#7
@step
def data_splitter(df:pd.DataFrame)->Tuple[Annotated[pd.DataFrame, 'x_train'],
                                        Annotated[pd.DataFrame, 'x_test'],
                                        Annotated[pd.Series, 'y_train'],
                                        Annotated[pd.Series, 'y_test']]:
    y = df.pop('PAYMENT_STATUS')
    x_train, x_test, y_train, y_test = train_test_split(df, y,test_size=0.25, random_state=42)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    return x_train, x_test, y_train, y_test

#8
@step
def model_selection(model_name:str)->Annotated[ClassifierMixin, 'Classification model selection']:
    if model_name.lower() == "rfc":
        model = RandomForestClassifier()
    elif model_name.lower() == 'abc':
        model = AdaBoostClassifier()
    else:
        raise ValueError
    return model

#9
@step
def model_training(model:ClassifierMixin, x_train:pd.DataFrame, y_train:pd.Series)->Annotated[ClassifierMixin, 'Trained model']:
    model.fit(x_train, y_train)
    return model

#10
@step
def model_prediction(model:ClassifierMixin, x_test:pd.DataFrame)->Annotated[pd.Series, 'MCC Predictions']:
    predictions = model.predict(x_test)
    predictions = pd.Series(predictions)
    return predictions

#11
@step
def model_evaluation(y_pred:pd.Series, y_test:pd.Series)->Tuple[Annotated[np.ndarray, 'Confusion Matrix'],
                                                                Annotated[(str | dict), 'Classification Report']]:
    confu_matrix = confusion_matrix(y_pred=y_pred, y_true=y_test)
    report = classification_report(y_pred=y_pred, y_true=y_test)
    print(f"""CONFUSION MATRIX
          f{confu_matrix}
            """)
    print(f"""CLASSIFICATION REPORT
          {report}
            """)
    return confu_matrix, report


    
