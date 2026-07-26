import pandas as pd
import numpy as np
import math

def mean(x):
    x=list(x)
    m=0
    for i in range(len(x)):
        m=m+x[i]
    return (m/len(x))

def variance(x):
    x=list(x)
    var=0
    m=mean(x)
    for i in range(len(x)):
        var=var+ ((x[i]-m)**2)
    var=var/len(x)
    std=math.sqrt(var)
    return var, std

def self_statistics(p):
    for col in p.columns:
        print(col,":")
        print("Mean: ",mean(p[col]))
        var, std=variance(p[col])
        print(f"Variance: {var:.3f}, S.D: {std:.3f}")
        print()

def numpy_statistics(p):
    print("Numpy Mean: ")
    print(np.mean(p, axis=0))
    print("Numpy Standard Deviation: ")
    print(np.std(p, axis=0))


p=pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
p=p.dropna()
p=p.drop(columns=['ID', 'Education', 'Marital_Status', 'Dt_Customer'])
self_statistics(p)
print("Compare with inbuilt Numpy functions")
numpy_statistics(p)
