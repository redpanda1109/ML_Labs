import pandas as pd
import numpy as np
from A1 import calculate_entropy
from A3_4 import dataset, binning, calc_info_gain


def build_decision_tree(X, Y, min_samples=5, depth=0, max_depth=5):
    # trying to build a tree i recursion where calc the best feature and split
    # find next best feature and split again until stopping criteria is met
    # All samples belong to one class
    if len(Y.unique()) == 1:
        return Y.iloc[0]
    # Maximum depth reached
    if depth >= max_depth:
        return Y.mode()[0]
    # No features left
    if len(X.columns) == 0:
        return Y.mode()[0]
    # Too few samples
    if len(Y) < min_samples:
        return Y.mode()[0]

    gains = {}
    for column in X.columns:
        gains[column] = calc_info_gain(X, Y, column)    #gives best feature per recursion
    best = max(gains, key=gains.get)

    # No useful split
    if gains[best] == 0:
        return Y.mode()[0]

    tree = {}
    tree[best] = {}
    for value in X[best].unique():
        # the dataset is split into based based on the values in that feature
        X_part = X[X[best] == value]
        Y_part = Y[X[best] == value]
        X_part = X_part.drop(columns=[best])
        tree[best][value] = build_decision_tree(X_part,Y_part,min_samples,depth + 1,max_depth)  #recursive here
    return tree


def main():
    X, Y = dataset("train_labels.csv")
    X = binning(X, bins=4)

    tree = build_decision_tree(X, Y)
    print("Decision Tree:")
    print(tree)


if __name__ == "__main__":
    main()