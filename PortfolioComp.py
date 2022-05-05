# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:23:21 2022

@author: evely
"""

#Evelyn Baranski (evelynb@bu.edu)
#4/20/22
#Assignment 13: Efficient portfolios, VaR, and drawdown

#This assignment, assignment 13 task 2, working different portfolios
# and calculating the risk and return characteristics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from MinVarPort_EfficientPort import *
from PortConstructionRebalance import *

from ModelVaR import compute_historical_var_pct, compute_model_var_pct
from DrawdownMCSim import *



## Portfolio madeup of 5 stocks of choosing for random time horizon

### compare investing stocks in different ways
# 1. equal weighted portfolio of the stocks
# 2. a mean-variance efficient portfolio of stocks that would achieve same 
### return as equally-weighted


#for each investment portfolio
## w/o rebalancing
## with rebalancing every 20 trading days


#descriptive stats and graphs to compare each portfolio and computng
## 1. 10 day value at risk (98th percentile)
## 2. the investors maximum drawdown from the highest level


## Equally weighted portfolio
def equal_nrb(symbols, initial_val = 1000 ):
    """This function is creating equally weighted  non rebalanced portfolio with symbols given.
    
    parameters
    symbols represents a list of 2 or more stock symbols in a list that are csv files
    optional parameter, initial_val, is set to 1,000 but can be chosen by user"""
    
    #getting prices 
    prices = get_stock_prices_from_csv_files(symbols)
    equal_weighted_df = create_equal_weight_portfolio(prices, initial_value = initial_val)
    
    return equal_weighted_df

    


def equal_rb(symbols, target_weights, initial_val = 1000):
    """This function is creating equally weighted  rebalanced portfolio with symbols given.
    
    parameters
    symbols represents a list of 2 or more stock symbols in a list that are csv files
    optional parameter, initial_val, is set to 1,000 but can be chosen by users
    
    rebalance - every 20 days
    """
    
    #getting prices 
    prices = get_stock_prices_from_csv_files(symbols)
    
    ##because equally weighted, weights are 20%
    equal_rebalance = rebalance_portfolio(prices, target_weights, 20, initial_val)
    
    return equal_rebalance




def efficient_nrb(symbols, initial_val = 1000):
    """This function is an efficient non rebalanced portfolio with symbols given.
    
    parameters
    symbols represents a list of 2 or more stock symbols in a list that are csv files
    optional parameter, initial_val, is set to 1,000 but can be chosen by users
    
    """
    #getting weights using returns, cov
    returns = get_stock_returns_from_csv_files(symbols)
    cov = get_covariance_matrix(returns)
    e = np.matrix(returns.mean())
    
    #getting equally weighted mean of return
    eq = equal_nrb(symbols, initial_val)
    eq = eq.pct_change()
    required_rate = eq["portfolio"].mean()
    
    v = np.matrix(cov)
    
    weights = calc_min_variance_portfolio(e, v, required_rate)
    weights = pd.DataFrame(weights)

    #getting prices, setting initial to weights
    prices = get_stock_prices_from_csv_files(symbols)
    prices[0:len(prices.columns)] = weights
    
    for x in range(1, len(prices)):
        prices.iloc[x] = prices.iloc[x - 1] * (1 + returns.iloc[x])
    
    #creating portfolio value - sum of asset columns
    prices["portfolio"] = prices.sum(axis=1)
    
    prices = prices * initial_val
    
    return prices



def efficient_rb(symbols, initial_val = 1000):
    """This function is an efficient rebalanced portfolio with symbols given.
    
    parameters
    symbols represents a list of 2 or more stock symbols in a list that are csv files
    optional parameter, initial_val, is set to 1,000 but can be chosen by users
    
    rebalance - every 20 days
    """
    
    ##getting weight, using returns covariance
    returns = get_stock_returns_from_csv_files(symbols)
    cov = get_covariance_matrix(returns)
    e = np.matrix(returns.mean())
    
    eq = equal_nrb(symbols, initial_val)
    eq = eq.pct_change()
    required_rate = eq["portfolio"].mean()
    
    v = np.matrix(cov)
    
    #using required rate from mean portfolio return
    weights = calc_min_variance_portfolio(e, v, required_rate)
    weights = weights.tolist()
    

    weight_target = {returns.columns[idx] : weights[0][idx] for idx in range(len(weights[0]))}
    prices = get_stock_prices_from_csv_files(symbols)
    
    eff_rb = rebalance_portfolio(prices, weight_target, 20, initial_val)
      
    return eff_rb
    




def compare(eq_no_rebalance, eq_rebalance, efficient_no_rebalance, efficient_rebalance):
    """This function is to compare each of the portfolios, generate descriptive
    statsitics and plotting relative weights over time of each asset within
    the portfolio"""
    
    print("We are going to compare the different portfolios")
    
    print()
    print("Equally weighted portfolio without rebalancing: ")
    print(eq_no_rebalance.pct_change().describe())
    print()
    
    print("Equally weighted portfolio with rebalancing: ")
    print(eq_rebalance.pct_change().describe())
    print()
    
    print("Mean variance portfolio without rebalancing: ")
    print(efficient_no_rebalance.pct_change().describe())
    print()
    
    print("Mean variance portfolio with rebalancing: ")
    print(efficient_rebalance.pct_change().describe())
    
    plot_relative_weights_over_time(eq_no_rebalance)
    plot_relative_weights_over_time(eq_rebalance)
    plot_relative_weights_over_time(efficient_no_rebalance)
    plot_relative_weights_over_time(efficient_rebalance)
    
    
    #getting 10-day VaR (98th percentile) for portfolio and drawdown P1
    portfolio = eq_no_rebalance
    returns = portfolio / portfolio.shift(1) - 1
    ret = returns["portfolio"].mean()
    daily_stdev = returns["portfolio"].std()
    
    var = compute_model_var_pct(ret, daily_stdev, .98, 10)
    
    dd = compute_drawdown(portfolio["portfolio"])
    max_dd = dd["dd_pct"].max()
    print("Equally weighted portfolio without rebalance VaR:", var, \
          "max drawdown:", max_dd)
    
        
    print()
    #P2
    portfolio = eq_rebalance
    returns = portfolio / portfolio.shift(1) - 1
    ret = returns["portfolio"].mean()
    daily_stdev = returns["portfolio"].std()
    
    var = compute_model_var_pct(ret, daily_stdev, .98, 10)
    
    dd = compute_drawdown(portfolio["portfolio"])
    max_dd = dd["dd_pct"].max()
    print("Equally weighted portfolio with rebalance VaR:", var, \
          "max drawdown:", max_dd)
    
        
    print()
    #P2
    portfolio = efficient_no_rebalance
    returns = portfolio / portfolio.shift(1) - 1
    ret = returns["portfolio"].mean()
    daily_stdev = returns["portfolio"].std()
    
    var = compute_model_var_pct(ret, daily_stdev, .98, 10)
    
    dd = compute_drawdown(portfolio["portfolio"])
    max_dd = dd["dd_pct"].max()
    print("Min variance portfolio without rebalance VaR:", var, \
          "max drawdown:", max_dd)
        
        
    print()
    #P2
    portfolio = efficient_rebalance
    returns = portfolio / portfolio.shift(1) - 1
    ret = returns["portfolio"].mean()
    daily_stdev = returns["portfolio"].std()
    
    var = compute_model_var_pct(ret, daily_stdev, .98, 10)
    
    dd = compute_drawdown(portfolio["portfolio"])
    max_dd = dd["dd_pct"].max()
    print("Min variance portfolio with rebalance VaR:", var, \
          "max drawdown:", max_dd)
    
    


         
        


if __name__ == '__main__':
    
    
    #Stocks I chose to use, all represent csv files in folder
    symbols = ['GOOG', 'JPM', 'IBM', 'FB', 'GM']
    
    equal_nrb_df = (equal_nrb(symbols))
    
    # 20% since evenly split
    target_weights = {'GOOG': 0.2, 'JPM':0.2, 'IBM':0.2, 'FB':0.2, 'GM':0.2}
    
    equally_rb = (equal_rb(symbols, target_weights, 1000))
    
    eff_nrb = (efficient_nrb(symbols))
    #print(eff_nrb)
    
    eff_rb = (efficient_rb(symbols))
    #print(eff_rb)
    
    compare(equal_nrb_df, equally_rb, eff_nrb, eff_rb)
    
    
    
