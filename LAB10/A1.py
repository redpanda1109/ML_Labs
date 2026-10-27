# combine mutliple models into one model using stacking
# use stacking classifier
# base models - rf, decision tree, adaboost
# meta model - logistic regression, svm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def data_preprocessing(p):
    mode_columns = ["gender","education","corpus"]
    for column in mode_columns:
        p[column] = p[column].fillna(p[column].mode()[0])
    # too many missing values so i cant determine what to replace with...so "unknown"
    unknown_columns = ["race","language","handedness"]
    for column in unknown_columns:
        p[column] = p[column].fillna("Unknown")
    
    Y=p['label']
    Y = Y.map({"CONTROL": 0,"MCI": 1,"ADRD": 2})
    X=p.drop(["uid","label","split","hash","filesize_kb", "corpus"],axis=1) #remove unecessary features
    X = pd.get_dummies(X,columns=["gender", "race", "language", "handedness", "education"])
    return X, Y


def main():
    p = pd.read_csv("train_labels.csv")
    X, Y = data_preprocessing(p)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

    base_models = [
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
        ("ada", AdaBoostClassifier(random_state=42))
    ]
    meta_models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
        ("SVM", SVC())
    ]

    for name, meta_model in meta_models:
        model = StackingClassifier(estimators=base_models,final_estimator=meta_model,cv=3)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)

    print("Accuracy: ", accuracy_score(Y_test, predictions))
    print("Precision: ",precision_score(Y_test, predictions, average='weighted'))
    print("Recall: ",recall_score(Y_test, predictions, average='weighted'))
    print("F1 Score: ",f1_score(Y_test, predictions, average='weighted'))

if __name__ == "__main__":
    main()