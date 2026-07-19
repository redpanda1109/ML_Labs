import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalization(p):
    p = p.replace('?', np.nan)
    numerics = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    for col in p.columns:
        if col in numerics:
            p[col] = p[col].fillna(p[col].median())
        elif col not in ['Record ID', 'Condition', 'referral source']:
            p[col] = p[col].map({'t': 1, 'f': 0, 'M': 0, 'F': 1}).fillna(0)

    normal = MinMaxScaler()
    p[numerics] = normal.fit_transform(p[numerics])
    print(p)
    
p=pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')
normalization(p)