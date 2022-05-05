# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:50:00 2022

@author: evely
"""

#Evelyn Baranski
#3/29/22
#Assignment 10: Efficient Portfolios


#This file, a10task2 is working on efficient portfolios
#and the efficient frontier


#importing necessary packages
import numpy as np
import math
from numpy.linalg import inv
import pandas as pd
import datetime as dt



#Function 1
def calc_portfolio_return(e, w):
    """This function calculates and returns the portfolio 
    return (as a float) for a portfolio of n >= 2 assets.
    
    e = matrix of expected returns for the assets
    w = matrix of portfolio weights of the asset (sum to 1)
    
    E[rp] = e * Wt"""
    
    #calculating - returning as float val
    ret = e * w.transpose()
    ret = float(ret)
    
    return ret


#Function 2
def calc_portfolio_stdev(v, w):
    """This function calculates and returns the portfolio
    standard deviation as a float for a portfolio of n >= 2 assets.
    v = matrix of covariances among assets
    w = matrix of portfolio weights
    
    stdev(rp) = sqrt(w * v * wT) """
    
    
    #getting standard deviation -- square root of variance
    std = w * v * w.transpose()
    std = float(math.sqrt(std))

    return std


#Function 3
def calc_global_min_variance_portfolio(v):
    """This function returns portfolio weights corresponding to
    the global minimum variance portfolio. Function will find
    the portfolio weights with the abs min variance that can be made with
    the assets given.
    
    v = matrix of covariances among assets
    
    c = 1t * v-1 * 1
    std (best) = 1 / c
    wp = std(best) * 1t * v-1
    
    1 = column vector of 1s same length as v
    std (best) = var min portfolio (scalar)"""
    
    #column vector of 1's the length of v
    ones = [1 for x in range(len(v))]
    ones = np.matrix(ones)
    ones = ones.transpose()
    
    c = ones.transpose() * inv(v)  * ones
    
    #best standard dev
    mvp = 1 / c
    wp = mvp * ones.transpose() * inv(v)
    
    return wp



#Function 4
def calc_min_variance_portfolio(e, v, r):
    """This function finds and returns the portfolio
    weights corresponding to the minimum variance
    portfolio for the required rate of r. """
    
    #column vector of 1's length of v
    ones = [1 for x in range(len(v))]
    ones = np.matrix(ones)


    #getting values for a, b, and c
    a = float(ones * inv(v) * e.transpose())
    b = float(e * inv(v) * e.transpose())
    c = float(ones * inv(v) * ones.transpose())

    
    #getting A value -- matrix and d, determinant of A
    A = np.matrix([[b, a], [a, c]])
    d = np.linalg.det(A)

    
    #getting g value and h value
    g = 1/d * (b * ones - a * e) * inv(v)
    h = 1/d * (c * e - a * ones) * inv(v)
    
    
    #vector of portfolio weights of the minimum
    #variance portfolio corresponding to expected rate of return, r
    wp = g + h * r

    return wp    


#function 5
def calc_efficient_portfolios_stdev(e, v, rs):
    """This function finds a series of minimum variance portfolios
    and returns their standard deviations using loop and accumulator
    pattern.
    
    e = matrix expected returns
    v = matrix of covariances of assets
    rs = numpy array of rates of return """
    
    #empty list for devs
    deviations = []
    
    #for loop to iterate through length of rates given for each
    #value rs
    for x in range(len(rs)):
        
        
        #setting rate to value rs[x]
        rate = rs[x]
        
        #getting weights, return, stdev
        w = calc_min_variance_portfolio(e, v, rate)
        r = calc_portfolio_return(e, w)       
        sigma = calc_portfolio_stdev(v, w)
        
        #appending stdev to sigma list
        deviations.append(sigma)
        
        #printing current r, sigma, and weights
        print(f"r = {r:.4f}, sigma = {sigma:.4f}, w = {w}")


    #deviation list into numpy array
    deviations = np.array(deviations)
    return deviations
        

#Function 6
def get_stock_prices_from_csv_files(symbols):
    """This function obtains a pd dataframe with historical
    stock prices for several stocks within the list symbols.
    Returns pd dataframe with monthly stock prices for each symbol
    for period of dates given within the csv"""
    
    
    # iterating through each symbol in list of symbols
    for symbol in symbols:
        
        #if statement for 1st symbol -- starting the file
        if symbol == symbols[0]:
            
            #processing and reading the csv, selecting columns, changing 'close'
            #to symbol name
            fn = './%s.csv' % symbol
            main = pd.read_csv(f'{fn}')
            main = main[['Date', 'Adj Close']]
            main.columns = ['Date', symbol]
        
        else:
            
            #proessing for other symbols, getting just adjusted close column
            fn = './%s.csv' % symbol
            df_current = pd.read_csv(f'{fn}') 
            df_current = df_current[['Adj Close']]
            df_current.columns = [symbol]
        
            #joining the two dataframes
            main = main.join(df_current)
        
    
    #making it monthly and setting date to the index
    main.set_index('Date', inplace=True)
    main.index = pd.to_datetime(main.index)
    main.resample('1M')
          
    return main
        


#Function 7
def get_stock_returns_from_csv_files(symbols):
    """This function returns a pd df containing stock return values
    from a list of symbols -- csv files for stock values"""
    
    
    #getting list of stock values
    stock_df = get_stock_prices_from_csv_files(symbols)
    
    
    #making df values numeric
    stock_df.apply(pd.to_numeric)
    
    #using pct_change to get returns
    returns = stock_df.astype(float).pct_change()
    
    return returns

    
    


#Function 8
def get_covariance_matrix(returns):
    """This function generates a covariance matrix for the
    stock returns in returns."""
    
    #using pd cov
    cov_matrix = returns.cov()
    
    return cov_matrix

    
    
            





if __name__ == '__main__':
    # e = np.matrix([0.1, 0.11, 0.08])
    # w = np.matrix([1,1,1]) / 3
    
    # print(calc_portfolio_return(e, w))
    
    # e = np.matrix([0.12, 0.05, 0.09])
    # w = np.matrix([0.3, 0.4, 0.3])
    # print(calc_portfolio_return(e, w))
    
    # w = np.matrix([0.4, 0.3, 0.3])
    # v = np.matrix([[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.15]])
    # print(calc_portfolio_stdev(v, w))
    
    # v = np.matrix([[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.15]])
    # w = calc_global_min_variance_portfolio(v)
    # print(w)

    # e = np.matrix([0.1, 0.11, 0.08])
    # v = np.matrix([[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.15]])
    # w = calc_min_variance_portfolio(e, v, 0.09)
    # print(w)
    
    # print(calc_portfolio_return(e, w))
    
    # e = np.matrix([0.12, 0.11, 0.08])
    # v = np.matrix([[0.3, 0.2, 0.1], [0.2, 0.25, 0.1], [0.1, 0.1, 0.2]])
    # w = calc_min_variance_portfolio(e, v, 0.09)
    # print(w)
    
    e = np.matrix([0.1, 0.11, 0.08])
    v = np.matrix([[0.2, 0, 0], [0, 0.1, 0], [0, 0, 0.15]])
    rs = np.linspace(0.07, 0.12, 10)
    
    sigmas = calc_efficient_portfolios_stdev(e, v, rs)
    
    print(sigmas)
    
    symbols = ['AAPL',  'DIS', 'GOOG', 'KO', 'WMT']
    
    returns = get_stock_returns_from_csv_files(symbols)
    
    cov = get_covariance_matrix(returns)
    
    print(cov)
    
    v = np.matrix(cov)
    e = np.matrix(returns.mean())
    
    w = calc_global_min_variance_portfolio(v)
    print(w)
    
    print(calc_portfolio_return(e, w))
    
    print(calc_portfolio_stdev(v, w))
    
