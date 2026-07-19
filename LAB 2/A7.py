import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(p):
    p = p.drop(columns=['Record ID', 'Condition', 'referral source'])
    p = p.replace('?', np.nan)
    numerics = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    for col in p.columns:
        if col in numerics:
            p[col] = p[col].fillna(0)
        else:
            p[col] = p[col].map({'t': 1, 'f': 0, 'M': 0, 'F': 1}).fillna(0)

    vectors=p.iloc[:20]
    binary=[c for c in vectors.columns if c not in numerics]
    b = vectors[binary].values
    total = vectors.values
    jc = np.zeros((20, 20))
    smc = np.zeros((20, 20))
    cos = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            f11 = np.sum((b[i] == 1) & (b[j] == 1))
            f00 = np.sum((b[i] == 0) & (b[j] == 0))
            f10 = np.sum((b[i] == 1) & (b[j] == 0))
            f01 = np.sum((b[i] == 0) & (b[j] == 1))
            jc[i][j] = f11 / (f11 + f10 + f01)
            smc[i][j] = (f11 + f00) / (f11 + f10 + f01 + f00)

            dot = np.dot(total[i], total[j])
            magA = np.sqrt(np.sum(np.square(total[i])))
            magB = np.sqrt(np.sum(np.square(total[j])))
            cos[i][j] = dot / (magA * magB)
    
    sns.heatmap(jc, annot=True)
    plt.show()
    sns.heatmap(smc, annot=True)
    plt.show()
    sns.heatmap(cos, annot=True)
    plt.show()

p=pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')
plot_heatmap(p)
