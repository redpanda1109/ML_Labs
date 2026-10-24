import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report


# Function to load the dataset
def load_dataset():
    data = pd.read_csv("train_labels(7).csv")
    features = [col for col in data.columns if col.startswith("MFCC_")]
    features.append("age")
    X = data[features]
    y = data["label"]
    return X, y


# Function to split the dataset
def split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42,stratify=y)
    return X_train, X_test, y_train, y_test


# Function to create the base models
def create_base_models():
    svm = Pipeline([("scaler", StandardScaler()),("classifier", SVC())])
    rf = RandomForestClassifier(random_state=42)
    dt = DecisionTreeClassifier(random_state=42)

    base_models = [("svm", svm),("rf", rf),("dt", dt)]
    return base_models


# Function to evaluate a model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(accuracy, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    return accuracy


# Function to test individual base models
def test_base_models(base_models, X_train, X_test, y_train, y_test):
    results = []

    for name, model in base_models:
        print("\nModel:", name)
        accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append({"Model": name,"Accuracy": accuracy})
    return results


# Function to implement stacking with different meta-models
def stacking_classifier(base_models, X_train, X_test, y_train, y_test):
    meta_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)  }

    results = []
    print("\n Stacking Classifier")
    for name, meta_model in meta_models.items():
        print("\nMeta-model:", name)
        model = StackingClassifier(estimators=base_models,final_estimator=meta_model,cv=5)
        accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append({
            "Model": "Stacking + " + name,
            "Accuracy": accuracy
        })
    return results


# Function to display the comparison
def display_results(base_results, stacking_results):
    results = base_results + stacking_results
    results_df = pd.DataFrame(results)
    print("\nComparison: ")
    print(results_df.to_string(index=False))


# Main function
def main():
    # Load dataset
    X, y = load_dataset()
    print("Dataset shape:", X.shape)
    print("\nClass distribution:")
    print(y.value_counts())
    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))
    # Create base models
    base_models = create_base_models()
    # Evaluate individual models
    base_results = test_base_models(base_models, X_train, X_test, y_train, y_test)
    # Evaluate stacking models
    stacking_results = stacking_classifier(base_models, X_train, X_test, y_train, y_test)
    # Compare results
    display_results(base_results, stacking_results)


if __name__ == "__main__":
    main()
