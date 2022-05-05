# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:04:00 2022

@author: evely
"""

#Evelyn Baranski (evelynb@bu.edu)
#4/20/22
#Assignment 13: Portfolio construction and rebalancing

#This assignment, assignment 13 task 1, working on portfolio
#construction and rebalancing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_equal_weight_portfolio(df_prices, initial_value = 1):
    """This function takes in parameters df_prices, a pd df
    containing historical price data and an optional param
    initial_value.
    
    Function returns a pd df with columns containing the values
    standardized to $1 or the initial value of an equally weighted
    investment in each of the assets in prices, as well as the value
    of the total portfolio without rebalancing."""
    
    #getting number of assets (columns) within the dataframe
    number_assets = len(df_prices.columns)
    
    #creating a df of percentage change in assets
    returns = df_prices.pct_change()
    
    #creating port copy of df_prices and creating initial weights
    port = df_prices
    port[0:number_assets-1] = initial_value / number_assets
    
    
    #iterating through to generate weight changes
    for x in range(1, len(port)):
        port.iloc[x] = port.iloc[x-1] * (1 + returns.iloc[x])
    
    #creating portfolio value - sum of asset columns
    port["portfolio"] = port.sum(axis=1)
        
    return port



def plot_relative_weights_over_time(df_values):
    """This function creates a plot showing the relative weights
    of each asset in the portfolio over time. 
    
    parameter df_values is a pd df containing columns of the assets
    values"""
    
    #plotting
    x = df_values.index
    
    port = df_values["portfolio"]
    df_values = df_values.drop(columns = "portfolio")
    
    #going through to plot each column (asset)
    for col in df_values.columns:
        y = df_values[col] / port
        
        plt.plot(x, y, label = col)
    
    plt.xlabel("Date")
    plt.legend()
    plt.show()
    
    
    
    
def rebalance_portfolio(df_prices, target_weights, rebalance_freq, initial_value = 1):
    """This function implements a portfolio rebalancing algorithm.
    
    df_prices = pd dataframe of daily historical asset prices
    target_weigthts = dictionary of target weighting in each asset
    rebalance_freq = frequency of rebalancing in trading day
    initial_value = optional parameter, set to 1
    
    function returns pd df with colmns contain values in each asset
    according to weights given with rebalancing done as required
    by rebalance_freq along with total value of portfolio"""
    
    
    
    #creating a df of percentage change in assets
    returns = df_prices.pct_change()
    returns = returns.fillna(0)
    
    #creating port copy of df_prices and creating initial weights
    port = df_prices.copy()
    
    #making values equal to target weights
    for x in port.columns:
        port[x] = target_weights[x] 
    
    
    #creating portfolio value - sum of asset columns
    port["portfolio"] = 1
    

    #iterating through length and columns to get asset value and total port value
    for x in range(len(port)):
        
        #if x != 0, calculating total portfolio value
        if x != 0:
            port["portfolio"].iloc[x] = sum(port[df_prices.columns].iloc[x - 1]  * (1 + returns[df_prices.columns].iloc[x])) 

        ## going through columns - if rebalance day = to port x weight, else equal to return previous
        for col in df_prices.columns:
            if x % rebalance_freq == 0:
                port[col].iloc[x] = port["portfolio"].iloc[x] * port[col].iloc[0]

            else:
                port[col].iloc[x] = port[col].iloc[x - 1] * (1 + returns[col].iloc[x])
        
        
    port = port.multiply(initial_value)
            
    return port




if __name__ == '__main__':
    
    prices = pd.read_csv('prices2.csv')
    
    prices.set_index('Date', inplace=True)

    
    
    print(create_equal_weight_portfolio(prices))
    
    values = create_equal_weight_portfolio(prices)
    
    returns = values / values.shift(1) - 1
    print(returns.describe())
    
    plot_relative_weights_over_time(values)
    
    #prices.index = prices["Date"]
    
    prices = pd.read_csv('prices2.csv')
    
    prices.set_index('Date', inplace=True)
    
    target_weights = {'KO': 0.6, 'GM':0.4}
    rebalance_freq = 2
    values = rebalance_portfolio(prices, 
                                target_weights, 
                                rebalance_freq, 1)
    
    
    plot_relative_weights_over_time(values)
    print(values)
    
    
    
    
    
    
