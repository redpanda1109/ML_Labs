import pandas as pd
import numpy as np
import math

def euclidean_norm(a,b):
    a=list(a)
    b=list(b)
    sum=0
    for i in range(len(a)):
        sum=sum+((a[i]-b[i])**2)
    sum=math.sqrt(sum)
    return sum

def nearest_centroid(point, centroids):
    min=euclidean_norm(point, centroids[0])
    cluster=0
    for i in range(1, len(centroids)):
        dist=euclidean_norm(point, centroids[i])
        if dist<min:
            min=dist
            cluster=i
    return cluster

def mean(cluster):
    centroid=[]
    for i in range(len(cluster[0])):
        total=0
        for j in range(len(cluster)):
            total=total+cluster[j][i]
        centroid.append(float(total/len(cluster)))
    return centroid


def kmeans(data, k):
    centroids=[]
    for i in range(k):
        centroids.append(data[i])
    while True:
        cluster=[]
        for i in range(k):
            cluster.append([])
        for points in data:
            i=nearest_centroid(points, centroids)
            cluster[i].append(points)
        new_centroid=[]
        for c in cluster:
            new_centroid.append(mean(c))
        if centroids == new_centroid:
            break
        centroids=new_centroid
    return centroids, cluster


p=pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
p=p.dropna()
p = p.drop(columns=['ID','Education','Marital_Status','Dt_Customer'])
features = p.columns
data=[]
for i in range(len(p)):
    s=[]
    for feature in features:
        s.append(p.iloc[i][feature])
    data.append(s)

k = int(input("Enter the number of clusters (K): "))
centroids, cluster = kmeans(data, k)
for i in range(len(centroids)):
    print("Cluster", i+1)
    print("Centroids: ",centroids[i])
    print("No.of points in cluster: ",len(cluster[i]))
    print()
