# LIME tells which features influenced the models decision
# tells if the feautures are positively or negatively influencing the models decision

import pandas as pd
from A2 import data_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer

def main():
    p = pd.read_csv("train_labels.csv")
    X, Y = data_preprocessing(p)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

    # pipeline for scaling and classification
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    model.fit(X_train, Y_train)
    
    # predict one value
    trial = X_test.iloc[0]
    prediction = model.predict(X_test.iloc[[0]])[0]     #double bracket to get the dataframe
    print("Actual Class:", Y_test.iloc[0])
    print("Predicted Class:", prediction)

    # Create a LIME explainer
    # train data, understand the feature names, type of model, final class names
    explainer = LimeTabularExplainer(X_train.values, feature_names=list(X_train.columns), mode='classification', class_names=['CONTROL', 'MCI', 'ADRD'])
    
    # Explain a prediction for a specific instance which is trial
    # what sample to explain, the prediction probabilities, and how many most imp features that affect
    explanation = explainer.explain_instance(trial.values, model.predict_proba, num_features=5)

    print("LIME Explanation:")
    print(explanation.as_list())

if __name__ == "__main__":
    main()
