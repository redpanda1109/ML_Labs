import pandas as pd
import numpy as np
import math

def euclidean_dist(a,b):
    a=list(a)
    b=list(b)
    sum=0
    for i in range(len(a)):
        sum=sum+((a[i]-b[i])**2)
    sum=math.sqrt(sum)
    return sum

def nearest_centroid(point, centroids):
    min=100000 
    nearest=0
    for i in range(1, len(centroids)):
        dist=euclidean_dist(point, centroids[i])    #compare the dist for each centroid
        if dist<min:
            min=dist
            nearest=i
    return nearest

# to find new centroids
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
    # lets take the first k points as the centroids initially and then update from there
    for i in range(k):
        centroids.append(data[i])
    # the loop runs till the centroids dont change 
    while True:
        cluster=[]
        for i in range(k):
            cluster.append([])  #empty cluster....so cluster will be like 3d, first cluster no., then the point
        for points in data:
            i=nearest_centroid(points, centroids)   #for every data find the nearest centroid
            cluster[i].append(points)   #the data is assigned to the cluster u found to be near
        new_centroid=[]
        for c in cluster:
            new_centroid.append(mean(c))
        if centroids == new_centroid:
            break   #convergence
        centroids=new_centroid
    return centroids, cluster


p=pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
p=p.dropna()
p = p.drop(columns=['ID','Education','Marital_Status','Dt_Customer'])
features = p.columns
data=[]
# id gives feature vector
for i in range(len(p)):
    s=[]
    for feature in features:
        s.append(p.iloc[i][feature])
    data.append(s)

k = int(input("Enter the number of clusters (K): "))
centroids, cluster = kmeans(data, k)
#for every cluster
for i in range(len(centroids)):
    print("Cluster", i+1)
    print("Centroids: ",centroids[i])
    print("No.of points in cluster: ",len(cluster[i]))
    print()
