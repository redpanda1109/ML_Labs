#4500 rows -> 2000 & 2500
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 

#stratify makes sure the ratio is same in train & test
#random_state gives the same split everytime
def data_split(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=42)
    return X_train, X_test, y_train, y_test


p=pd.read_excel('thyroid_dataset.xlsx')
X=p.drop(columns=['Condition'])
Y=p['Condition']
print(Y.value_counts())
X_train, X_test, y_train, y_test = data_split(X, Y)
print(y_train.value_counts())
print(y_test.value_counts())
