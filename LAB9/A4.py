import pandas as pd
from A2 import data_preprocessing
from A3 import performance_metrics, random_forest_classify, decision_tree_classify, perceptron_classify, mlp_classify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline #imp one

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def model_pipeline():
    classifiers = [("Decision Tree", decision_tree_classify), ("Random Forest", random_forest_classify),
                ("Perceptron", perceptron_classify), ("MLP", mlp_classify)]

    # for each classifier we create a pipeline
    pipelines = []
    for name, classify in classifiers:
        # pipeline means its a sequence of steps that run simultaneously
        model=Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classify(None, None, None, None))
        ])
        pipelines.append((name, model))
    return pipelines

def main():
    p = pd.read_csv("train_labels.csv")
    X, Y = data_preprocessing(p)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

    results=[]
    pipelines = model_pipeline()
    for name, model in pipelines:
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        accuracy, precision, recall, f1 = performance_metrics(Y_test, predictions)
        results.append((name, accuracy, precision, recall, f1))

    print("Model Performance Metrics:")
    for name, accuracy, precision, recall, f1 in results:
        print(f"{name}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")

if __name__ == "__main__":
    main()


