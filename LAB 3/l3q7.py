import math
import pandas as pd
import numpy as np

def dot_product(a,b,n):
    a=list(a)
    b=list(b)
    sum=0
    for i in range(n):
        sum+= a[i]*b[i]
    return sum

def euclidean_norm(a,n):
    a=list(a)
    sum=0
    for i in range(n):
        sum += a[i]*a[i]
    sum=math.sqrt(sum)
    return sum

A = [10, 11, 14, 15]
B = [16, 18, 24, 79]
n=len(A)
dot=dot_product(A, B, n)
euclidean_A=euclidean_norm(A, n)
print("Self Dot product: ",dot)
print("Numpy Dot Product: ",np.dot(A,B))
print("Self Euclidean Norm A: ",euclidean_A)
print("Numpy Euclidean Norm A: ",np.linalg.norm(A))
