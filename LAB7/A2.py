import pandas as pd
import numpy as np

def calculate_gini(y):
    #give diff y values
    count=y.value_counts()
    gini=0
    for c in count:
        prob=c/len(y)
        gini+=prob**2
    # gini=1-sigma(prob^2)
    gini=1-gini
    return gini


def main():
    p=pd.read_csv('train_labels.csv')
    y=p['label']
    print(y.value_counts())
    print(y.value_counts(normalize=True))
    print("Gini Index of Dataset: ", calculate_gini(y))

if __name__ == "__main__":
    main()