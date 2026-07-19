import numpy as np
import pandas as pd

def cosine_sim(p):
    p = p.drop(columns=['Record ID', 'Condition', 'referral source'])
    p = p.replace('?', np.nan)
    numerics = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    for col in p.columns:
        if col in numerics:
            p[col] = p[col].fillna(0)
        else:
            p[col] = p[col].map({'t': 1, 'f': 0, 'M': 0, 'F': 1}).fillna(0)

    A = np.array(p.loc[0], dtype=float)
    B = np.array(p.loc[1], dtype=float)
    dot = np.dot(A, B)
    magA = np.sqrt(np.sum(np.square(A)))
    magB = np.sqrt(np.sum(np.square(B)))
    cos = dot / (magA * magB)
    return cos

p = pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')
result = cosine_sim(p)
print("Cosine Similarity:", result)