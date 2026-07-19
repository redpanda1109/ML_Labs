import numpy as np
import pandas as pd

def ranges(p):
    for col in p.columns:
        if col in numerics:
            min=p[col].min()
            max=p[col].max()
            print(f"Range of {col}: min: {min} - max: {max}")

def missing_values(p):
    for col in p.columns:
        missing=p[col].isnull().sum()
        print(f"Missing values in {col}: {missing}")

def mean_variance(p):
    for col in numerics:
        mean=p[col].mean()
        variance=p[col].var()
        print(f"Mean of {col}: {mean}, Variance: {variance}")

p=pd.read_excel('Lab Session Data.xlsx', sheet_name='thyroid0387_UCI') 
p = p.replace('?', np.nan)
numerics = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']

ranges(p)
missing_values(p)
mean_variance(p)