import pandas as pd
from A2 import data_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB

def performance_metrics(Y_test, predictions):
    accuracy = accuracy_score(Y_test, predictions)
    precision = precision_score(Y_test, predictions, average='weighted')
    recall = recall_score(Y_test, predictions, average='weighted')
    f1 = f1_score(Y_test, predictions, average='weighted')
    return accuracy, precision, recall, f1

def svm_classify(X_train, Y_train, X_test, Y_test):
    model = SVC(random_state=42)
    return model

def decision_tree_classify(X_train, Y_train, X_test, Y_test):
    model = DecisionTreeClassifier(random_state=42)
    return model

def random_forest_classify(X_train, Y_train, X_test, Y_test):
    model = RandomForestClassifier(random_state=42)
    return model

def catboost_classify(X_train, Y_train, X_test, Y_test):
    model = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
    return model

def adaboost_classify(X_train, Y_train, X_test, Y_test):
    model = AdaBoostClassifier(random_state=42)
    return model

def xgboost_classify(X_train, Y_train, X_test, Y_test):
    model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    return model

def naive_bayes_classify(X_train, Y_train, X_test, Y_test):
    model = GaussianNB()
    return model

def main():
    p=pd.read_csv('train_labels.csv')
    X, Y = data_preprocessing(p)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
    #normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results=[]
    classifiers = [("SVM", svm_classify), ("Decision Tree", decision_tree_classify), ("Random Forest", random_forest_classify),
        ("CatBoost", catboost_classify), ("AdaBoost", adaboost_classify), ("XGBoost", xgboost_classify), ("Naive Bayes", naive_bayes_classify)]
    for name, classify in classifiers:
        model=classify(X_train, Y_train, X_test, Y_test)
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        accuracy, precision, recall, f1 = performance_metrics(Y_test, predictions)
        results.append((name, accuracy, precision, recall, f1))

    for i in results:
        print(f"{i[0]}: Accuracy={i[1]:.4f}, Precision={i[2]:.4f}, Recall={i[3]:.4f}, F1 Score={i[4]:.4f}") 


if __name__ == "__main__":
    main()