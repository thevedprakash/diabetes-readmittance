import pandas as pd
import pickle
import joblib
import os
from sklearn.model_selection import train_test_split
from preprocessing import Preprocess, Preprocessing
import warnings
warnings.filterwarnings("ignore")
import pickle
import joblib
import os
from sklearn.model_selection import train_test_split
from model import Logistic_regression



def save_model(model, file_name):
    joblib.dump(model, file_name)


def save_pickle(model, file_name):
    with open(file_name, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(file_name):
    model = joblib.load(file_name)
    return model




def train(X, y, modelType):
    # Split your dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = modelType(X_train, X_test, y_train, y_test)
    return model


if __name__ == "__main__":
    path = '/home/ris/pythonProject/diabetes-readmittance/data/diabetic_train.csv'

    target = 'readmitted'





    X, y, transform_dict = Preprocessing(target, path)

    with open('/home/ris/pythonProject/diabetes-readmittance/models/transform_dict.pickle', 'wb') as handle:
        pickle.dump(transform_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Statrted Training the Linear_regression model.")
    Logistic_regression_model = train(X, y, Logistic_regression)
    print("Saving the model.")
    file_name = "/home/ris/pythonProject/diabetes-readmittance/models/Logistic_regression.pickle"
    save_pickle(Logistic_regression_model, file_name)

