from sklearn.model_selection import train_test_split
import pandas as pd



if __name__ == "__main__":
    path = '/home/ris/pythonProject/diabetes-readmittance/data/diabetes_raw.csv'
    df = pd.read_csv(path)
    train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    train.to_csv('/home/ris/pythonProject/diabetes-readmittance/data/diabetic_train.csv', index=False)
    test.to_csv('/home/ris/pythonProject/diabetes-readmittance/data/diabetic_test.csv', index=False)