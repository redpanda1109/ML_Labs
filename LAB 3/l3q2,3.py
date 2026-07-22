import pandas as pd
import numpy as np

def label_encoding(p,ordinal):
    p = p.replace('?', np.nan)
    types=list(p['Education'].unique())
    index=[]
    for i in range(len(types)):
        index.append(i)
    labelled=dict(zip(types,index))
    return labelled

def one_hot_encoding(p,nominal):
    p = p.replace('?', np.nan)
    for data in nominal:
        one_hot=pd.get_dummies(p[data], dtype=int)
        
        combined=pd.concat([data, one_hot], axis=1)
        p.to_excel()
    return p
        


p=pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
nominal = ['Marital_Status', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain','Response']
ordinal = ['Eucation']
print(label_encoding(p,ordinal))
print(one_hot_encoding(p, nominal))