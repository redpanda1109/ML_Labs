# AI tool used: ChatGPT for function creation and displaying the values
import pandas as pd
import numpy as np
import math


df = pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')

numerical_df = df.select_dtypes(include=np.number)
X = numerical_df.to_numpy(dtype=float)


def self_statistics(X):
    n = X.shape[0]
    mean = np.sum(X, axis=0) / n
    variance = np.sum((X - mean) ** 2, axis=0) / n
    standard_deviation = np.sqrt(variance)
    return mean, variance, standard_deviation


def comparison(X, self_mean, self_variance, self_std):
    package_mean = np.mean(X, axis=0)
    package_variance = np.var(X, axis=0)
    package_std = np.std(X, axis=0)

    print("COMPARISON")
    print(f"{'Feature':<20}"
          f"{'Self Mean':<15}"
          f"{'Package Mean':<15}"
          f"{'Self Var':<15}"
          f"{'Package Var':<15}"
          f"{'Self SD':<20}"
          f"{'Package SD':<20}"
          f"{'Difference'}")
    print("-" * 120)

    for i, feature in enumerate(numerical_df.columns):
        difference = abs(self_std[i] - package_std[i])
        print(f"{feature:<20}"
              f"{self_mean[i]:<15.6f}"
              f"{package_mean[i]:<15.6f}"
              f"{self_variance[i]:<15.6f}"
              f"{package_variance[i]:<15.6f}"
              f"{self_std[i]:<20.6f}"
              f"{package_std[i]:<20.6f}"
              f"{difference:.10f}")


mean, variance, standard_deviation = self_statistics(X)
print("RESULTS FROM self_statistics()")
print("\nMean:")
for feature, value in zip(numerical_df.columns, mean):
    print(f"{feature}: {value:.6f}")
print("\nVariance:")
for feature, value in zip(numerical_df.columns, variance):
    print(f"{feature}: {value:.6f}")
print("\nStandard Deviation:")
for feature, value in zip(numerical_df.columns, standard_deviation):
    print(f"{feature}: {value:.6f}")
print()

comparison(X, mean, variance, standard_deviation)