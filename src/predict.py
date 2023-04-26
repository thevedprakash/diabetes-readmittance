from preprocessing import Preprocess, Preprocessing
from variables import cols_to_be_dropped
import pickle
import joblib
import warnings
warnings.filterwarnings("ignore")



def encode_predict( df, transform_dict ):

    df = data.remove_cols(df, cols_to_be_dropped)
    df = data.scalling(df)
    df = data.remove_outliers(df)
    df = df.replace(transform_dict)
    X = data.filter_predictor_columns(df)
    return X

if __name__ == "__main__":
    with open('/home/ris/pythonProject/diabetes-readmittance/models/transform_dict.pickle', 'rb') as handle:
        transform_dict = pickle.load(handle)
    print('encoded dict test_data',transform_dict)


    model_path = "/home/ris/pythonProject/diabetes-readmittance/models/Logistic_regression.pickle"
    saved_model = joblib.load(model_path)

    path = '/home/ris/pythonProject/diabetes-readmittance/data/diabetic_test.csv'
    data = Preprocess(path)
    df = data.df
    test_input = encode_predict(df,  transform_dict)

    print(test_input.head())
    prediction = saved_model.predict(test_input)
    print(prediction)


