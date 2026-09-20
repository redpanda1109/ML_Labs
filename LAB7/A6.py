import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from A3_4 import dataset, binning

def main():
    X, Y = dataset("train_labels.csv")
    X = binning(X, bins=4)
    # do one-hot cause decision tree in sklearn cant deal with categorical data
    X = pd.get_dummies(X)
    Y = Y.map({"CONTROL": 0,"MCI": 1,"ADRD": 2})
    # criteria entropy is for the parameter for info gain
    model = DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=42)
    model.fit(X, Y)

    # Plot the tree
    plt.figure(figsize=(25, 15))
    tree.plot_tree(model,feature_names=X.columns,class_names=["CONTROL", "MCI", "ADRD"],filled=True,rounded=True,fontsize=7)
    plt.title("Decision Tree Visualization")
    plt.show()
    plt.savefig("A6_decision_tree.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()