# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:02:04 2022

@author: evely
"""

#Evelyn Baranski (evelynb@bu.edu)
#4/11/22
#Assignment 12: Quantifying Investment Risk


#This assignment, assignment 12 task 1, is on quantifying investment
#risk -- looking at VaR (Value at Risk)

from scipy.stats import norm
import math
import pandas as pd
import numpy as np



def compute_model_var_pct(mu, sigma, x, n):
    """This function is computing value at risk as a percentage
    of the asset/portfolio value. Returns float number.
    
    This VaR estimate uses model of the asset's returns assuming
    a normal distribution of daily returns.
    
    parameters:
    mu = mean daily rate of return
    sigma = daily standard deviation of retunrs
    x = percent confidence in maximum % loss over course of n days
    n = # of days"""
    
    #getting z value using x confidence level
    z = norm.ppf(1 - x)
    
    var = (mu * n) + z * sigma * math.sqrt(n)
    
    return float(var)



def compute_historical_var_pct(returns, x, n):
    """This function is compiuting VaR as a percentage using
    the historical simulation approach. 
    
    The parameter 'returns' a pandas series
    containing historical daily stock returns with a date index.
    
    x & n used to calibrate estimate: x is percent confidence
    of our maximum percentage loss over the course of n days
    
    """
    
    # #reading in CSV file
    # returns = pd.read_csv(returns)
    # # returns.index = returns["Date"]

    col = returns[1]
    
    #sorting pd series returns
    returns = returns.sort_values(ascending=False)

    
    #getting day and consequitive VaR
    day = math.floor(len(returns.index) * x)


    #day = returns[col].iloc[day]
    n_day_var = returns[day] * math.sqrt(n)
    
    
    #print output
    print(returns.describe())
    print(f"With compute_historical_var_pct estimate, we are {x*100:.0f}% \
confident that the maximum loss over {n} days would not exceed {n_day_var*100:.2f}%")

    return n_day_var
    
    
    





if __name__ == '__main__':
    print(compute_model_var_pct(.0008, .01, .98, 10))
    print(compute_model_var_pct(.001, .015, .97, 14))
    
    # #getting pandas series from csv file
    # df = pd.read_csv('SPY.csv')
    
    # df.index = pd.to_datetime(df['Date'])
    # prices = pd.DataFrame(df.loc['2021-01-01':'2021-12-31', 'Adj Close'])
    
    df = pd.read_csv('SPY.csv')
    df['ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    returns = df['ret']
    
    compute_historical_var_pct(returns, .98, 7)
    
