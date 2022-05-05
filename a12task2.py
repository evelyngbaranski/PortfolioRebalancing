# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:50:28 2022

@author: evely
"""

#Evelyn Baranski (evelynb@bu.edu)
#4/11/22
#Assignment 12: Quantifying Investment Risk


#This assignment, assignment 12 task 2, is on quantifying investment
#risk -- looking at drawdown


## drawdown is measure of loss of value using the msot recent maximum value
## until today. Drawdown is a way to quantify worst-case scenario


import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#importing from task 9 for Monte Carlo simulator
from a9task1 import MCStockSimulator




def compute_drawdown(prices):
    """This function processes an np.array object or pd series
    containing a column array of daily asset prices.
    
    Returns pd df containing columns
    price = copy of data in prices
    prev_max = previous peak price before this price
    dd_dollars = drawdown since maximum price measured in dollars
    dd_pct = percentage decline since prev_max"""
    
    #creating new df -- adding column for prices from prices
    dd_df = pd.DataFrame(index = prices.index)
    dd_df["price"] = prices
    
    #creating new variables
    dd_df["prev_max"] = 0
    dd_df["dd_dollars"] = 0
    dd_df["dd_pct"] = 0
    
    #creating current_max value
    current_max = dd_df["price"].iloc[0]
    
    #iterating through to assign previous max values
    for col in range(len(dd_df)):
        dd_df['prev_max'].iloc[col] = current_max
        
        #when price > current previous max value, taking its place
        if dd_df["price"].iloc[col] > current_max:
            current_max = dd_df["price"].iloc[col]
    
    
    #iterating through assigning values for dd_dollars & dd_pct
    for col in range(len(dd_df)):
    
        #when price < prev_max -- drawdown
        if dd_df['price'][col] < dd_df['prev_max'][col]:
            
            dd_df['dd_dollars'].iloc[col] = dd_df['prev_max'].iloc[col] - dd_df['price'].iloc[col]
            dd_df['dd_pct'].iloc[col] = (dd_df['prev_max'].iloc[col] - dd_df['price'].iloc[col]) / dd_df['prev_max'].iloc[col]
    
            
    
    return dd_df



def plot_drawdown(df):
    """This function creates 2 charts, the historical prices with the
    maxmimum price & the drawdown since previous maxmimum price as
    percentage lost 
    df paramters contains columns price, prev_max, dd_dollars, dd_pct
    """
    
    #Plotting historical price and previous maximum price
    fig1, ax1 = plt.subplots()
    
    y = df['price']
    x = df.index
    ax1.plot(x, y, label = "price")
    
    y2 = df['prev_max']
    ax1.plot(x, y2, label = 'prev_max')
    ax1.set_title('Price and Previous Maximum')
    ax1.set_xlabel('Date')
    ax1.legend()
    
    
    #Plotting drawdown since previous maxmimum price as percentage lost
    fig2, ax2 = plt.subplots()
    
    y = df['dd_pct']
    x = df.index
    ax2.plot(x, y, label = 'dd_pct')
    ax2.set_title('Drawdown Percentage')
    ax2.set_xlabel('Date')
    ax2.legend()
    
    


def run_mc_drawdown_trials(init_price, years, r, sigma, trial_size, num_trials):
    """This function uses Monte Carlo simulation to simulate the price path
    evolution of a stock. Will create an instance of MCStockSimulator and run
    num_trials and calculate the maxmimum drawdown for each trial
    
    parameters
    init_price = stock price @ time 0
    years = number of years to run simulation
    r = mean annual rate of r
    sigma = stdev of annual returns
    trial_size = num of discrete steps per year (trading days)
    num_trials = num trials to include in simulation run"""
    
    
    #creating object of MCStockSimulator class
    sim = MCStockSimulator(init_price, years, r, sigma, trial_size)


    #empty list to hold maximum values from each trial
    trials = []
    
    #for loop to perform each trial
    for x in range(num_trials):
        
        
        #simulating stock values and creating df of values
        values = sim.generate_simulated_stock_values()
        values = pd.DataFrame(values)
        
        #generating drawdown df of values and extracting dd_pct (pct change dd)
        draw = compute_drawdown(values)
        dd_df = draw["dd_pct"]
        
        #getting maximum drawdown value from simulation values 
        max_dd = dd_df.max()
        trials.append(max_dd)



    trials_series = pd.Series(trials)

    return trials_series
        
        
    
    
    
    

    

    
    


if __name__ == '__main__':
    df = pd.read_csv('GM.csv')
    
    df.index = pd.to_datetime(df['Date'])
    prices = pd.DataFrame(df.loc['2021-01-01':'2021-12-31', 'Adj Close'])
    
    dd = compute_drawdown(prices)
    print(dd)
    
    plot_drawdown(dd)
    
    
    # #part 2
    df = pd.read_csv('SPY.csv')
    prices = pd.DataFrame(df.loc['2021-01-01':'2021-12-31', 'Adj Close'])
    df['ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1)) 
    trial_size = 252
    init_price = float(df['Adj Close'].sample())
    r = df['ret'].mean() * trial_size
    sigma = df['ret'].std() * np.sqrt(trial_size)
    years = 10
    num_trials = 100
    
    max_dd = (run_mc_drawdown_trials(init_price, years, r, sigma, trial_size, num_trials))
    max_dd.describe()