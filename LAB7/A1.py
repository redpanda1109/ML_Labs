import pandas as pd
import numpy as np

def calculate_entropy(y):
    #gives all diff y values
    count=y.value_counts()
    entropy=0
    for c in count:
        prob=c/len(y)
        entropy+=prob*np.log2(prob)
    entropy=-entropy
    return entropy


def main():
    p=pd.read_csv('train_labels.csv')
    y=p['label']
    print(y.value_counts())
    print(y.value_counts(normalize=True))
    print("Entropy of Dataset: ", calculate_entropy(y))

if __name__ == "__main__":
    main()