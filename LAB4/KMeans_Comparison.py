import time
import numpy as np
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

from my_kmeans import (kmeans as my_kmeans,data_clean as my_data_clean)
from ai_kmeans import (kmeans as ai_kmeans)


def execution_time(kmeans_function, data, k):
    start_time = time.perf_counter()
    result = kmeans_function(data, k)
    end_time = time.perf_counter()
    return result, end_time - start_time


def calculate_wcss(clusters, centroids):
    wcss = 0
    for i in range(len(clusters)):
        for point in clusters[i]:
            for j in range(len(point)):
                wcss += (
                    point[j] - centroids[i][j]
                ) ** 2
    return wcss


def silhouette_score(data, clusters):
    labels = []
    for i in range(len(clusters)):
        for point in clusters[i]:
            labels.append(i)
    data_array = np.array(data)
    return sklearn_silhouette_score(
        data_array,
        labels
    )


def cluster_size(clusters):
    sizes = []
    for cluster in clusters:
        sizes.append(len(cluster))
    return sizes


def get_iterations(result):
    return result[2]


def compare_kmeans(data):
    for k in [4, 5]:
        print("\n" + "=" * 75)
        print("K =", k)
        print("=" * 75)
# My K-MEANS
        my_result, my_execution_time = execution_time(
            my_kmeans,
            data,
            k
        )
        my_centroids = my_result[0]
        my_clusters = my_result[1]
        my_iterations = get_iterations(
            my_result
        )
        my_wcss = calculate_wcss(
            my_clusters,
            my_centroids
        )
        my_silhouette = silhouette_score(
            data,
            my_clusters
        )
        my_cluster_sizes = cluster_size(
            my_clusters
        )

# AI K-MEANS
        ai_result, ai_execution_time = execution_time(
            ai_kmeans,
            data,
            k
        )
        ai_clusters = ai_result[0]
        ai_centroids = ai_result[1]
        ai_iterations = get_iterations(
            ai_result
        )
        ai_wcss = calculate_wcss(
            ai_clusters,
            ai_centroids
        )
        ai_silhouette = silhouette_score(
            data,
            ai_clusters
        )
        ai_cluster_sizes = cluster_size(
            ai_clusters
        )

        print("\nMetric                    My K-Means          AI K-Means")
        print("-" * 70)
        print(
            f"Execution Time            "
            f"{my_execution_time:.6f} s        "
            f"{ai_execution_time:.6f} s"
        )
        print(
            f"Iterations                "
            f"{my_iterations:<20} "
            f"{ai_iterations}"
        )
        print(
            f"WCSS                      "
            f"{my_wcss:<20.4f} "
            f"{ai_wcss:.4f}"
        )
        print(
            f"Silhouette Score          "
            f"{my_silhouette:<20.4f} "
            f"{ai_silhouette:.4f}"
        )
        print(
            f"Cluster Sizes             "
            f"{str(my_cluster_sizes):<20} "
            f"{ai_cluster_sizes}"
        )
        print("\nFinal Centroids - My K-Means:")
        for i, centroid in enumerate(my_centroids):
            print("Centroid", i + 1, ":", centroid)
        print("\nFinal Centroids - AI K-Means:")
        for i, centroid in enumerate(ai_centroids):
            print("Centroid", i + 1, ":", centroid)


if __name__ == "__main__":
    data, numerical_df = my_data_clean("Lab Session Data.xlsx","marketing_campaign")
    print("Dataset dimensions:")
    print(len(data), "rows x", len(data[0]), "features")
    print("\nRunning comparison for K = 4 and K = 5...")
    compare_kmeans(data)
