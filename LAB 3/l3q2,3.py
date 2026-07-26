import pandas as pd
import numpy as np

def label_encoding(p,ordinal):
    p = p.replace('?', np.nan)
    types=list(p['Education'].unique())
    index=[]
    for i in range(len(types)):
        index.append(i)
    labelled=dict(zip(types,index))
    p['Education']=p['Education'].replace(labelled)
    return p

def one_hot_encoding(p,nominal):
    p = p.replace('?', np.nan)
    for data in nominal:
        one_hot=pd.get_dummies(p[data], dtype=int)
        p=pd.concat([p, one_hot], axis=1)
        p=p.drop(data, axis=1)
    return p



p=pd.read_excel("Lab Session Data.xlsx", sheet_name='marketing_campaign')
nominal = ['Marital_Status', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Complain','Response']
ordinal = ['Eucation']
interval = ['Year_Brith', 'Dt_Customer']
p_new=label_encoding(p,ordinal)
print(p_new['Education'])
print(one_hot_encoding(p_new, nominal))
