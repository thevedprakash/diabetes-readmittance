from preprocessing import Preprocess, Preprocessing
from variables import cols_to_be_dropped
import pickle
import joblib
import warnings
warnings.filterwarnings("ignore")

def encode_predict_input(df, encoded_dict):
    object_columns = df.select_dtypes(object).columns
    label_dict = encoded_dict
    for col in object_columns:
        df[col] = df[col].replace(label_dict)
    return df


def encode_predict( path, target, encoded_dict ):
    data = Preprocess(path)
    df = data.df
    df = data.remove_cols(df, cols_to_be_dropped)
    df = data.scalling(df)
    df = data.remove_outliers(df)
    df = encode_predict_input(df, encoded_dict)
    X = df.drop(target, axis = 1)
    return X

if __name__ == "__main__":
    print("Loading the TestData.")
    # Load data (deserialize)
    with open('/home/ris/pythonProject/diabetes-readmittance/models/encoded.pickle', 'rb') as handle:
        encoded_dict = pickle.load(handle)

    print('encoded dict test_data',encoded_dict)


    model_path = "/home/ris/pythonProject/diabetes-readmittance/models/knearest.pickle"
    saved_model = joblib.load(model_path)
    path = '/home/ris/pythonProject/diabetes-readmittance/data/diabetic_test.csv'
    target = 'readmitted'

    test_input = encode_predict(path, target, encoded_dict)



    print(test_input.head())
    saved_model.predict(test_input)
    print(saved_model.predict(test_input))


