import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def package_mean_variance(p):
    prices=p['Price'].values
    m=np.mean(prices)
    v=np.var(prices)
    return m, v

def self_mean_variance(p):
    prices=p['Price'].values
    m= sum(prices) / len(prices)
    temp=0
    for i in prices:
        temp += (i - m) ** 2
    v= temp / len(prices)
    return m, v

def time_computation(p):
    st1=time.time_ns()
    for i in range(10):
        m1, v1 = package_mean_variance(p)
    et1=time.time_ns()
    avg1=(et1-st1)/10
    st2=time.time_ns()
    for i in range(10):
        m2, v2 = self_mean_variance(p)
    et2=time.time_ns()
    avg2=(et2-st2)/10
    return avg1, avg2

def wenesday(p):
    wed=p.loc[p['Day']=='Wed', 'Price']
    w=np.mean(wed)
    return w

def april(p):
    apr=p.loc[p['Month']=='Apr', 'Price']
    a=np.mean(apr)
    return a

def probability_of_loss(p):
    all=list(p['Chg%'])
    loss=list(filter(lambda x: x < 0, all))
    prob=len(loss)/len(all)
    return prob

def prob_on_wed(p):
    total = len(p.loc[(p['Day'] == 'Wed') & (p['Chg%'] > 0)])
    prob = total / len(p.loc[p['Day'] == 'Wed'])
    return prob 

def plot_scatter(p):
    plt.scatter(p['Day'], p['Chg%'])
    plt.title('Chg% vs Day')
    plt.xlabel('Day')
    plt.ylabel('Chg%')
    plt.show()

p=pd.read_excel('Lab Session Data.xlsx', sheet_name='IRCTC Stock Price')
p['Chg%'] = p['Chg%'].astype(str).str.rstrip('%').astype(float)
m1, v1 = package_mean_variance(p)
m2, v2 = self_mean_variance(p)
print('Python package mean: ', m1, ' variance: ', v1)
print('Self method mean: ', m2, ' variance: ', v2)
avg1, avg2 = time_computation(p)
print('Average time taken by python package(in ns): ', avg1)
print('Average time taken by self method(in ns): ', avg2)

wed_mean = wenesday(p)
print('Mean price on Wednesday: ', wed_mean)
if wed_mean > m1:
    print('Mean price on Wednesday is greater than population mean')
else:
    print('Mean price on Wednesday is less than population mean')

apr_mean = april(p)
print('Mean price in April: ', apr_mean)
if apr_mean > m1:
    print('Mean price in April is greater than population mean')
else:
    print('Mean price in April is less than population mean')

prob = probability_of_loss(p)
print('Probability of loss: ', prob)
prob_wed = prob_on_wed(p)
print('Probability of profit on Wednesday: ', prob_wed)
print('Conditional probability of making profit, given that today is Wednesday: ', prob_wed/prob)

plot_scatter(p)