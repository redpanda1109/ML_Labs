import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

def minkowski_dist(u,v,order):
    temp=0
    u=list(u)
    v=list(v)
    for i in range(len(u)):
        temp+=math.pow(abs(u[i]-v[i]),order)
    result=math.pow(temp,(1/order))
    return result

def minkowski_dist_vectors(p):
    feature_vectors=['Income', 'Recency']
    p=p[feature_vectors].dropna()
    x=p[feature_vectors[0]]
    y=p[feature_vectors[1]]
    d=[]
    for i in range(1,11):
        dist=minkowski_dist(x, y, i)
        d.append(dist)
    plt.plot(range(1, 11), d)
    plt.xlabel('p')
    plt.ylabel('Dist')
    plt.show()

def compare(p,k):
    feature_vectors=['Income', 'Recency']
    p=p[feature_vectors].dropna()
    x=p[feature_vectors[0]]
    y=p[feature_vectors[1]]
    dist=minkowski_dist(x, y, k)
    package=distance.minkowski(x, y, k)
    return dist, package

u=[10, 11, 14, 15]
v=[16, 18, 24, 79]
print("Distance between u & v using Manhattan Dist: ", minkowski_dist(u,v,1))
print("Distance between u & v using Euclidean Dist: ", minkowski_dist(u,v,2))

p=pd.read_excel('Lab Session Data.xlsx', sheet_name='marketing_campaign')
p=p.dropna()

minkowski_dist_vectors(p)
for i in range(1,4):
    x,y=compare(p,i)
    print("Power= ",i)
    print("My dist: ",x)
    print("Scipy Dist: ",y)





