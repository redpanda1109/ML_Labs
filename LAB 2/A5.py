import numpy as np
import pandas as pd

def similarity(p):
    col=[]
    for cols in p.columns:
        uniques=p[cols].dropna().unique()
        if set(uniques).issubset({'t','f'}):
            col.append(cols)
    row1=p.loc[0, col]
    row2=p.loc[1, col]
    r1=np.array([1 if i=='t' else 0 for i in row1])
    r2=np.array([1 if i=='t' else 0 for i in row2])

    f11 = np.sum((r1 == 1) & (r2 == 1))
    f00 = np.sum((r1 == 0) & (r2 == 0))
    f10 = np.sum((r1 == 1) & (r2 == 0))
    f01 = np.sum((r1 == 0) & (r2 == 1))
    jc= f11 / (f11 + f10 + f01)
    sm= (f11 + f00) / (f11 + f10 + f01 + f00)
    return jc, sm



p=pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI') 
jc, sm = similarity(p)
print("Jaccard Coefficient:", jc)
print("Simple Matching Coefficient:", sm)
