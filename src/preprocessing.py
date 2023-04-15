import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from variables import cols_to_be_dropped
import warnings
warnings.filterwarnings("ignore")
from outliers import remove_race_outlier, remove_discharge_outlier, remove_readmitted_outlier , remove_gender_outlier, remove_admission_type_outlier, remove_admission_source_id_outlier

class Preprocess:

  def __init__(self, path):
    self.df = pd.read_csv(path)
    print("data uplaoded!")


  def remove_cols(self, df,  cols_to_dropped):
    df = df.drop(cols_to_dropped, axis = 1)
    print('columns dropped sucessfully!')
    return df

  def scalling(self, df):
      num_columns = df.select_dtypes(exclude=["object"]).columns
      normal = StandardScaler()
      df[num_columns] = normal.fit_transform(df[num_columns])
      print('Scalling of numericals columns done!')
      return df


  def remove_outliers(self, df):
      df['readmitted'] = df['readmitted'].apply(remove_readmitted_outlier)
      df['admission_source_id'] = df['admission_source_id'].apply(remove_admission_source_id_outlier)
      df['discharge_disposition_id'] = df['discharge_disposition_id'].apply(remove_discharge_outlier)
      df['admission_type_id'] = df['admission_type_id'].apply(remove_admission_type_outlier)
      df['gender'] = df['gender'].apply(remove_gender_outlier)
      df['race'] = df['race'].apply(remove_race_outlier)
      print('Handling of categorical variables done!')
      return df
  def handling_cat(self, df):
    object_columns = df.select_dtypes(object).columns
    for col in object_columns:
        le = LabelEncoder()
        le.fit(df[col])
        encoded_dict = dict(zip((le.classes_), le.transform(le.classes_)))
        df[col] = df[col].replace(encoded_dict)
    print('encoding of categorical columns done!')
    return df, encoded_dict


def Preprocessing(df, target, path):
    data = Preprocess(path)
    df = data.df
    df = data.remove_cols(df, cols_to_be_dropped)
    df = data.scalling(df)
    df = data.remove_outliers(df)
    df, encoded_dict = data.handling_cat(df)
    X = df.drop(target, axis = 1)
    y = df[target]
    return X, y, encoded_dict



if __name__ == "__main__":
    path = '/home/ris/pythonProject/diabetes-readmittance/data/diabetes_raw.csv'

    target = 'readmitted'
    data = Preprocess(path)
    df = data.df

    X, y, encoded_dict = Preprocessing(df, target)



