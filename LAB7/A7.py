import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from A3_4 import dataset

def main():
    X, Y = dataset("train_labels.csv")
    X = X[["MFCC_1_mean", "age"]]   # ramdomly picked an 2 features
    Y = Y.map({"CONTROL": 0,"MCI": 1,"ADRD": 2})    #label encoding

    # Create decision tree
    model = DecisionTreeClassifier(criterion="entropy",max_depth=5,random_state=42)
    model.fit(X, Y)

    # Plot decision boundary
    plt.figure(figsize=(10, 7))
    disp=DecisionBoundaryDisplay.from_estimator(model,X,response_method="predict",xlabel="MFCC_1_mean",ylabel="age",alpha=0.5)
    # Plot actual data points
    # the plot will have background color for the decision boundary and the actual data points will be plotted on top of it
    # u have to observe that most points will match with the background color
    scatter=plt.scatter(X["MFCC_1_mean"],X["age"],c=Y,cmap=ListedColormap(disp.multiclass_colors_),edgecolor="black",s=20)
    plt.title("Decision Boundary using MFCC_1_mean and age")
    plt.figlegend(scatter.legend_elements()[0],["CONTROL", "MCI", "ADRD"],loc="lower center",ncols=3)
    plt.savefig("A7_decision_boundary.png",dpi=300,bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()