import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from variables import cols_to_be_dropped, predictor_columns
from outliers import *
import warnings
warnings.filterwarnings("ignore")

class Preprocess:

  def __init__(self, path):
    self.df = pd.read_csv(path)
    print("data uplaoded!")


  def remove_cols(self, df,  cols_to_dropped):
    df = df.drop(cols_to_dropped, axis = 1)

    print('Irrevelent columns dropped sucessfully!')
    return df

  def scalling(self, df):
      num_columns = df.select_dtypes(exclude=["object"]).columns
      normal = StandardScaler()
      df[num_columns] = normal.fit_transform(df[num_columns])

      print('Scalling of numericals columns done!')
      return df


  def remove_outliers(self, df):
      df['race'] = df['race'].apply(remove_race_outlier)
      df['gender'] = df['gender'].apply(remove_gender_outlier)
      df['admission_type_id'] = df['admission_type_id'].apply(remove_admission_type_outlier)
      df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(remove_discharge_outlier)
      df['admission_source_id'] = df['admission_source_id'].apply(remove_admission_source_id_outlier)
      df['readmitted'] = df['readmitted'].apply(remove_readmitted_outlier)

      print('Outliers removed sucessfully!')
      return df


  def mapping_dict(self, df):
      object_columns = df.select_dtypes(object).columns

      transform_dict = {}
      for col in object_columns:
          cats = pd.Categorical(df[col]).categories
          d = {}
          for i, cat in enumerate(cats):
              d[cat] = i
          transform_dict[col] = d
      print('mapping of encoding dictionary done!')
      return transform_dict, df


  def handling_cat(self, df):
    object_columns = df.select_dtypes(object).columns
    print(object_columns)
    for col in object_columns:
        le = LabelEncoder()
        le.fit(df[col])
        encoded_dict = dict(zip((le.classes_), le.transform(le.classes_)))
        df[col] = df[col].replace(encoded_dict)
    print('encoding of categorical columns done!')

    return df

  def filter_predictor_columns(self, df):
    filter_predictor_columns = predictor_columns
    return df[filter_predictor_columns]





def Preprocessing(target, path):
    data = Preprocess(path)
    df = data.df
    df = data.remove_cols(df, cols_to_be_dropped)
    df = data.scalling(df)
    df = data.remove_outliers(df)
    transform_dict, df = data.mapping_dict(df)
    df= data.handling_cat(df)
    X = data.filter_predictor_columns(df)
    y = df[target]
    return X, y, transform_dict



if __name__ == "__main__":
    path = '/home/ris/pythonProject/diabetes-readmittance/data/diabetic_train.csv'
    target = 'readmitted'

    X, y, transform_dict = Preprocessing(target, path)
    print(transform_dict)








