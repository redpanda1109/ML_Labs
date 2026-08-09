import pandas as pd
import numpy as np
import random


def data_clean(filename, name):
    df = pd.read_excel(filename, sheet_name=name)
    df = df.drop_duplicates()
    df = df.drop(columns=['ID','Education','Marital_Status','Dt_Customer'])
    numerical_df = df.select_dtypes(include=np.number)
    numerical_df = numerical_df.fillna(numerical_df.median())
    data = []
    for row in numerical_df.itertuples(index=False):
        data.append(list(row))
    return data, numerical_df


def distance(point, centroid):
    distance_value = 0
    for i in range(len(point)):
        distance_value += (point[i] - centroid[i]) ** 2
    distance_value = distance_value ** 0.5
    return distance_value


def kmeans(data, k):
    centroids = random.sample(data, k)
    max_iterations = 100
    for iteration in range(max_iterations):
        clusters = [[] for _ in range(k)]

        for point in data:
            distances = []
            for centroid in centroids:
                d = distance(point, centroid)
                distances.append(d)
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)

        new_centroids = []
        for i in range(k):
            if len(clusters[i]) == 0:
                new_centroids.append(centroids[i])
            else:
                cluster_array = np.array(clusters[i])
                new_centroid = cluster_array.mean(axis=0).tolist()
                new_centroids.append(new_centroid)

        if np.allclose(centroids, new_centroids):
            print("\nK-Means converged after", iteration + 1, "iterations.")
            centroids = new_centroids
            break
        centroids = new_centroids
    return clusters, centroids


data, numerical_df = data_clean("Lab Session Data.xlsx", 'marketing_campaign')

print("Numerical Features:")
print(len(numerical_df.columns))
print(list(numerical_df.columns))
print("Data Matrix Dimensions:", len(data), "x", len(data[0]))
print()

k = random.randint(3, 7)
print("\nRandomly selected K:", k)
clusters, centroids = kmeans(data, k)
print("K-MEANS RESULTS")
print()
for i in range(k):
    print("\nCluster", i + 1)
    print("Number of points:", len(clusters[i]))
    print("Centroid:")
    print(centroids[i])