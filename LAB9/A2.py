#lets use the train_label.csv as the dataset for this
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

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

def perceptron_random_search(X_train, Y_train):
    model = Perceptron(random_state=42)
    # penalty - regularization, alpha - multiplied value with regularization, max_iter - no.of iterations, eta0 - learning rate
    # using some 4 hyperparameters for the random search to tune the model
    parameters = {"penalty": [None, "l2", "l1", "elasticnet"],"alpha": [0.0001, 0.001, 0.01, 0.1],"max_iter": [1000, 2000, 3000],
                "eta0": [0.01, 0.1, 1]}
    random_search = RandomizedSearchCV(model, parameters, n_iter=10, cv=5, scoring="accuracy", random_state=42)
    random_search.fit(X_train, Y_train)
    return random_search

def mlp_random_search(X_train, Y_train):
    model = MLPClassifier(random_state=42)
    # hidden_layer_sizes - no.of neurons in each layer, activation - activation function, max_iter - no.of iterations,
    # alpha - multiplied value with regularization, learning_rate_init - learning rate
    parameters = {"hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],"activation": ["relu", "tanh", "logistic"],
                "max_iter": [300, 500, 1000],"alpha": [0.0001, 0.001, 0.01],"learning_rate_init": [0.001, 0.01, 0.1]}
    random_search = RandomizedSearchCV(model, parameters, n_iter=10, cv=5, scoring="accuracy", random_state=42)
    random_search.fit(X_train, Y_train)
    return random_search

def main():
    p=pd.read_csv('train_labels.csv')
    X, Y = data_preprocessing(p)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
    #normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    random_search_p = perceptron_random_search(X_train, Y_train)
    predictions = random_search_p.predict(X_test)
    print("PERCEPTRON")
    print("Best Parameters:")
    print(random_search_p.best_params_)
    print("Best Accuracy:")
    print(random_search_p.best_score_)
    print("Test Accuracy: ")
    print(accuracy_score(Y_test, predictions))

    random_search_mlp = mlp_random_search(X_train, Y_train)
    predictions = random_search_mlp.predict(X_test)
    print("MLP")
    print("Best Parameters:")
    print(random_search_mlp.best_params_)
    print("Best Accuracy:")
    print(random_search_mlp.best_score_)
    print("Test Accuracy: ")
    print(accuracy_score(Y_test, predictions))


if __name__ == "__main__":
    main()
