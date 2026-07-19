import numpy as np
import pandas as pd

def data_imputation(p):
    p=p.replace('?', np.nan)
    numeric_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
    for col in p.columns:
        if col in numeric_cols:
            median_val=p[col].median()
            p[col]=p[col].fillna(median_val)
        elif col not in ['Record ID', 'Condition', 'referral source']:
            mode_val=p[col].mode()[0]
            p[col]=p[col].fillna(mode_val)
    print('Data imputation is done')
    print(p)


p=pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI')
data_imputation(p)