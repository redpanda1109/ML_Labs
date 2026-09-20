import pandas as pd
import numpy as np
from A1 import calculate_entropy

def dataset(file_path):
    p=pd.read_csv(file_path)
    # checks for all features with missing values
    missing = p.isnull().sum()
    print(missing)
    # less missing values so fill with mode
    mode_columns = ["gender","education","corpus"]
    for column in mode_columns:
        p[column] = p[column].fillna(p[column].mode()[0])
    # too many missing values so i cant determine what to replace with...so "unknown"
    unknown_columns = ["race","language","handedness"]
    for column in unknown_columns:
        p[column] = p[column].fillna("Unknown")

    Y=p['label']
    X=p.drop(["uid","label","split","hash","filesize_kb", "corpus"],axis=1) #remove unecessary features
    return X, Y

def equal_width_binning(column, bins):
    #grouping the values into a fixed range
    mini=column.min()
    maxi=column.max()
    bin_width=(maxi-mini)/bins
    result=[]
    for i in column:
        idx = int((i - mini) / bin_width)   #the whole no. part gives the bin
        if idx == bins:  # handles when value equals max
            idx -= 1
        result.append(idx)
    return result

def binning(X, bins=4):
    # do binning to all numeric features
    numerical = ["age", "MFCC_1_mean", "MFCC_2_mean", "MFCC_3_mean", "MFCC_4_mean", "MFCC_5_mean", "MFCC_6_mean", "MFCC_7_mean", 
            "MFCC_8_mean", "MFCC_9_mean", "MFCC_10_mean", "MFCC_11_mean", "MFCC_12_mean", "MFCC_13_mean", "MFCC_1_std", "MFCC_2_std", 
            "MFCC_3_std", "MFCC_4_std", "MFCC_5_std", "MFCC_6_std", "MFCC_7_std", "MFCC_8_std", "MFCC_9_std", "MFCC_10_std", 
            "MFCC_11_std", "MFCC_12_std", "MFCC_13_std"]
    for column in numerical:
        X[column] = equal_width_binning(X[column], bins)
    return X

def calc_info_gain(X, Y, column):
    total_entropy = calculate_entropy(Y)
    weighted = 0
    values = X[column].unique()
    for value in values:
        value_Y = Y[X[column] == value]     # for a given feature find the prob and calculate the entropy for that subset of data
        prob = len(value_Y) / len(Y)
        entropy = calculate_entropy(value_Y)
        weighted += prob * entropy
    info_gain = total_entropy - weighted
    return info_gain

def best_feature(X, Y):
    #for each feature and the one with max info gain is the best one
    info_gains = {}
    for column in X.columns:
        info_gains[column] = calc_info_gain(X, Y, column)
    best=max(info_gains, key=info_gains.get)
    return best, info_gains[best]

def main():
    X,Y=dataset('train_labels.csv')
    X=binning(X, bins=4)
    best, info_gain = best_feature(X, Y)
    print("Best Feature: ", best)
    print("Information Gain: ", info_gain)

if __name__ == "__main__":
    main()