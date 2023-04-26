import pickle
import sys, os
sys.path.append("src")
sys.path.append("models")

from variables import *
from outliers import *
from sklearn.preprocessing import StandardScaler





def remove_cols(df,  cols_to_dropped):
    df = df.drop(cols_to_dropped, axis = 1)
    return df

def scalling(df):
    num_columns = df.select_dtypes(exclude=["object"]).columns
    normal = StandardScaler()
    df[num_columns] = normal.fit_transform(df[num_columns])
    return df

def remove_outliers(df):
    df['race'] = df['race'].apply(remove_race_outlier)
    df['gender'] = df['gender'].apply(remove_gender_outlier)
    #df['admission_type_id'] = df['admission_type_id'].apply(remove_admission_type_outlier)
    df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(remove_discharge_outlier)
    df['admission_source_id'] = df['admission_source_id'].apply(remove_admission_source_id_outlier)
    #df['readmitted'] = df['readmitted'].apply(remove_readmitted_outlier)
    return df

def filter_predictor_columns(df):
    filter_predictor_columns = predictor_columns
    return df[filter_predictor_columns]

def encode_predict(df, encoded_dict):
    df = scalling(df)
    df = remove_outliers(df)
    df = df.replace(encoded_dict)
    X = filter_predictor_columns(df)
    return X
