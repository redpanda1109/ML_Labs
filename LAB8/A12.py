import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    # Load project dataset
    data = pd.read_csv("train_labels.csv")

    # Fill missing values
    data["gender"] = data["gender"].fillna(data["gender"].mode()[0])
    data["education"] = data["education"].fillna(data["education"].mode()[0])
    data["corpus"] = data["corpus"].fillna(data["corpus"].mode()[0])
    data["race"] = data["race"].fillna("Unknown")
    data["language"] = data["language"].fillna("Unknown")
    data["handedness"] = data["handedness"].fillna("Unknown")

    # Remove columns that are not useful for prediction
    data = data.drop(["uid", "hash", "split", "filesize_kb"],axis=1)

    # Convert categorical columns into numbers
    data = pd.get_dummies(data,columns=["gender", "race", "language", "handedness", "education", "corpus"])

    # Separate input and target
    X = data.drop("label", axis=1)
    y = data["label"]

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    # Normalize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create MLP model
    model = MLPClassifier(hidden_layer_sizes=(10, 10),activation="relu",solver="adam",learning_rate_init=0.001,max_iter=500,random_state=42)
    model.fit(X_train, y_train)    # Train the model
    predictions = model.predict(X_test)   #predcit the model
    accuracy = accuracy_score(y_test, predictions)  #accuracy

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))
    print("Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()