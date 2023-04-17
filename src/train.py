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
from models import regression, knearestneighbour, decisiontree



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





    X, y, encoded_dict = Preprocessing(target, path)

    with open('/home/ris/pythonProject/diabetes-readmittance/models/encoded.pickle', 'wb') as handle:
        pickle.dump(encoded_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Statrted Training the Linear_regression model.")
    regression_model = train(X, y, regression)
    print("Saving the model.")
    file_name = "/home/ris/pythonProject/diabetes-readmittance/models/regression.pickle"
    save_pickle(regression_model, file_name)

    print("Statrted Training the decisiontree model.")
    decisiontree_model = train(X, y, decisiontree)
    print("Saving the model.")
    file_name = "/home/ris/pythonProject/diabetes-readmittance/models/decision.pickle"
    save_pickle(decisiontree_model, file_name)

    print("Statrted Training the knearestneighbour model.")
    knearestneighbour_model = train(X, y, knearestneighbour)
    print("Saving the model.")
    file_name = "/home/ris/pythonProject/diabetes-readmittance/models/knearest.pickle"
    save_pickle(knearestneighbour_model, file_name)