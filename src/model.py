from sklearn import linear_model

from sklearn.linear_model import LogisticRegression


def Logistic_regression(X_train, X_test, y_train, y_test):
    logistic_reg = LogisticRegression(tol=1e-7, penalty='l2', C=0.0005)
    logistic_reg.fit(X_train, y_train)
    Ylog = logistic_reg.predict(X_test)
    print(" The accuracy of the Logistic regression model:", logistic_reg.score(X_test, y_test))
    return logistic_reg

