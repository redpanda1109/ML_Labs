import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from A1 import data_preprocessing

def main():
    p=pd.read_csv("train_labels.csv")
    X, Y = data_preprocessing(p)

    correlation = X.corr()
    print("Correlation Matrix:")
    print(correlation)

    sns.heatmap(correlation, annot=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()

if __name__ == "__main__":
    main()