import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from A3_4 import dataset, binning

def main():
    X, Y = dataset("train_labels.csv")
    X = binning(X, bins=4)
    X = pd.get_dummies(X)
    Y = Y.map({"CONTROL": 0,"MCI": 1,"ADRD": 2})
    model = DecisionTreeClassifier(random_state=42)
    parameters = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 9],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid = GridSearchCV(model,parameters,cv=5,scoring="accuracy")
    grid.fit(X, Y)
    print("GRID SEARCH")
    print("Best Parameters:")
    print(grid.best_params_)
    print("Best Accuracy:")
    print(grid.best_score_)

    random = RandomizedSearchCV(model,parameters,n_iter=20,cv=5,scoring="accuracy",random_state=42)
    random.fit(X, Y)
    print("\n\nRANDOM SEARCH")
    print("Best Parameters:")
    print(random.best_params_)
    print("Best Accuracy:")
    print(random.best_score_)


if __name__ == "__main__":
    main()

