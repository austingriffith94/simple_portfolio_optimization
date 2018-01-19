# Austin Griffith
# Simple Portfolio Optimization
# Python 3.6.3
# 1/17/2018

import pandas as pd
import numpy as np
from gurobipy import *
import math
import os
import matplotlib.pyplot as plt

# create directory for graphs
name = 'data'
if os.path.exists(name) == False:
    os.makedirs(name)

#%%
# read in monthly returns data
ret = pd.read_csv('monthly_returns.csv')
ret = ret.set_index('Unnamed: 0')
ret = ret.reset_index(drop=True)

# summary statistics of the data
avg = ret.mean()
std = ret.std()
cov = ret.cov()

#%%
# outputting average and std deviation of returns
# and covariance matrix
print('Average Returns')
print(avg)
print('\nStd Deviation of Returns')
print(std)
print('\nCovariance Matrix of Returns')
print(cov)

#%%
# create new model for the minimum risk portfolio
model = Model('min_risk')

# labels and variables for each stock
tickers = ret.columns
variables = pd.Series(model.addVars(tickers),index=tickers)

# determine the risk using the covariance matrix
port_risk = cov.dot(variables).dot(variables)

#%%
# set the model to minimize
model.setObjective(port_risk,GRB.MINIMIZE)

# constraints
# weights add up to 1
model.addConstr(variables.sum() == 1,'weights')
model.update()
# no shorting stocks(w >= 0)
model.setParam('OutputFlag',0)
model.update()

# optimize model, finds minimum risk portfolio with constraints
model.optimize()

#%%
# display variables and respective weights
n = 0
weights = {}
for v in variables:
    weights.update({tickers[n]:v.x})
    n = n + 1
weights = pd.DataFrame([weights])
weights = weights.transpose()
weights.columns = ['Weights']

print('\nMin Risk, Optimal Weights Per Stock')
print(weights['Weights'])

#%%
# organize dataframes
main = pd.concat([avg,std,weights],axis=1)
main.columns = ['Avg','Std','Weights']

# save values to csv
cov.to_csv(name+'/CovarianceRet.csv')
main.to_csv(name+'/MainRet.csv')

#%%
# minimum risk values
# optimal objective value
print('\nMinimized Portfolio Variance : '+str(port_risk.getValue()))
# volatility
min_vol = math.sqrt(port_risk.getValue())
print('Volatility : '+str(min_vol))
# expected return using optimized weights
port_return = avg.dot(variables)
Rmin = port_return.getValue()
print('Expected Return (Rmin) : '+str(Rmin))

#%%
# maximum return value among all stocks
Rmax = avg.max()

# return constraint
target = model.addConstr(port_return == Rmin, 'target')

# calculate values of efficient frontier
# set right hand side of target value for returns
# iterate through the range of returns from Rmin to Rmax
eff = {}
iterations = 50
diff = (Rmax-Rmin)/(iterations-1)
Rrange = np.arange(Rmin,Rmax+diff,diff)
for r in Rrange:
    target.rhs = r
    model.optimize()
    temp = math.sqrt(port_risk.getValue())
    eff.update({temp:r})

# organize dataframe for efficient frontier
frontier = pd.DataFrame([eff]).transpose()
frontier.columns = ['Returns']
frontier['Risk'] = frontier.index
frontier = frontier.reset_index(drop=True)

# output and save values of efficient frontier
print('\nEfficient Frontier')
print(frontier)
frontier.to_csv(name+'/EffFrontier.csv')

#%%
# plot of the efficient frontier from Rmin to Rmax
# initialize plot, set labels
fig, ax = plt.subplots(nrows=1,ncols=1)
fig.set_size_inches(16,9)
ax.set_title('Efficient Frontier of a Portfolio',fontsize=20)
ax.set_xlabel('Risk',fontsize=14)
ax.set_ylabel('Return',fontsize=14)

# plot the efficient frontier
# do this first to allow individual points later on to overlay
ax.scatter(x=frontier['Risk'],y=frontier['Returns'],color='orange',label='Efficient Frontier')
ax.plot(x=frontier['Risk'],y=frontier['Returns'],color='orange')
temp = pd.DataFrame([eff]).transpose()
temp.columns = ['Efficient Frontier']
temp.plot(color='orange',label='Efficient Frontier',ax=ax)

# average return/volatility for each individual stock
ax.scatter(x=std,y=avg,color='green',label='Stocks')
i = 0
for stock in tickers:
    ax.annotate(stock,(std[i],avg[i]))
    i = i + 1

# show the minimum risk portfolio
ax.scatter(x=min_vol,y=Rmin,color='blue',label='Optimal')
ax.annotate('Min. Risk',(min_vol,Rmin))

# additional edits to the graph
ax.grid()
ax.legend(loc='upper left')
fig.savefig(name+'/EfficientFrontier.png')
