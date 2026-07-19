from xml.parsers.expat import model

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
# this is a binary classification model where we can use logistic regression algorithm to determine rich/poor

def matrixes(p):
    X=p[['Candies (#)', 'Mangoes (Kg)','Milk Packets (#)']]
    Y=p[['Payment (Rs)']]
    return X, Y
def categorize(p, X, Y):
    # the model uses the logistic regression algorithm to classify customers
    model = LogisticRegression()
    # the training of the model is done using the fit method with X and Y as inputs
    model.fit(X, Y)
    p['Customer_Type'] = model.predict(X)

p=pd.read_excel('Lab Session Data.xlsx', sheet_name='Purchase data')    
X, Y = matrixes(p)
Y = np.where(p['Payment (Rs)'] > 200, 'RICH', 'POOR')
categorize(p, X, Y)
print(p[['Customer', 'Payment (Rs)', 'Customer_Type']])
