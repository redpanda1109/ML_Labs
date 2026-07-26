import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mean(x):
    x=list(x)
    sum=0
    for i in range(len(x)):
        sum=sum+x[i]
    return (sum/len(x))

def variance(x):
    x=list(x)
    var=0
    m=mean(x)
    for i in range(len(x)):
        var=var+((x[i]-m)**2)
    var=var/len(x)
    return var

def histogram(p):
    feature = "Income"
    x = p[feature].dropna()
    counts, bins = np.histogram(x, bins=10)
    print("Histogram Counts: ",counts)
    print("Histogram Bins: ",bins)

    plt.hist(x, bins=10)
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()

p=pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
print("Feature = Income")
x=p['Income'].dropna()
print("Mean: ",mean(x))
print("Variance: ",variance(x))
histogram(p)