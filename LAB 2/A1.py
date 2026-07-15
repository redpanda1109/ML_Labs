import numpy as np
import pandas as pd
from numpy.linalg import matrix_rank

def matrixes(p):
    X=p[['Candies (#)', 'Mangoes (Kg)','Milk Packets (#)']]
    Y=p[['Payment (Rs)']]
    return X, Y

def featureRank(X):
    rank=matrix_rank(X)
    return rank
def Cost(X,Y):
    inverse=np.linalg.pinv(X)
    x1=np.dot(inverse,Y)
    return x1

p=pd.read_excel('Lab Session Data.xlsx', sheet_name='Purchase data')
X, Y = matrixes(p)
print("Rank of X:", featureRank(X))
print("Cost of each product:", Cost(X, Y))

