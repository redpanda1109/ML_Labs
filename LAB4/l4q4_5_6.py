# AI tool used: ChatGPT for function creation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski


def minkowski_distance(x, y, p):
    """
    p = 1  -> Manhattan Distance
    p = 2  -> Euclidean Distance
    p > 2  -> Generalized Minkowski Distance
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    distance = np.sum(np.abs(x - y) ** p) ** (1 / p)
    return distance


def apply_minkowski(x, y):
    p_values = range(1, 11)
    distances = []
    for p in p_values:
        distance = minkowski_distance(x, y, p)
        distances.append(distance)
        print(f"Minkowski Distance (p={p}): {distance:.6f}")
    # Plot graph
    plt.figure(figsize=(8, 5))
    plt.plot(
        p_values,
        distances,
        marker="o"
    )
    plt.xlabel("Value of p")
    plt.ylabel("Minkowski Distance")
    plt.title("Minkowski Distance for p = 1 to 10")
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.show()
    return list(p_values), distances


def comparison(x, y, p_values, calculated_distances):
    print("\nComparison with scipy.spatial.distance.minkowski()")
    print("-" * 70)
    print(f"{'p':<5}{'Our Function':<20}{'Scipy Function':<20}{'Difference'}")
    print("-" * 70)
    for p, our_distance in zip(p_values, calculated_distances):
        scipy_distance = minkowski(x, y, p)
        difference = abs(our_distance - scipy_distance)
        print(
            f"{p:<5}"
            f"{our_distance:<20.6f}"
            f"{scipy_distance:<20.6f}"
            f"{difference:.10f}"
        )


df = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
data = df[["Income", "Recency"]].dropna()
vector1 = data["Income"].values
vector2 = data["Recency"].values

p_values, distances = apply_minkowski(vector1, vector2)

comparison(
    vector1,
    vector2,
    p_values,
    distances
)