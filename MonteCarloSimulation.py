# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 15:49:02 2022

@author: evely
"""

####Evelyn Baranski
# 3/20/22
#assingment 9: Monte Carlo Simulation and Pricing Exotic
# options


#This file, a9task1, is simulating stock returns


#importing numpy
import numpy as np
import matplotlib.pyplot as plt

import math



#Creating class

class MCStockSimulator:
    """Creating MCStockSimulator class"""
    
    def __init__(self, s, t, mu, sigma, nper_per_year):
        """Constructor to initialize the variables"""
        
        
        #initializing all of the parameters
        self.s = s
        self.t = t
        self.mu = mu
        self.sigma = sigma
        self.nper_per_year = nper_per_year
    
    
    def __repr__(self):
        """Creating repr method for the class"""
        
        
        #s string to be returned for formatting
        s = f"MCStockSimulator (s = ${self.s:.2f}, t = {self.t:.2f} (years), \
mu = {self.mu:.2f}, sigma = {self.sigma:.2f}, nper_per_year = {self.nper_per_year}"

        return s
    
    
    
    #Method generate_simulated_stock_returns
    def generate_simulated_stock_returns(self):
        """This method will generate and return an np.array containing
        a sequence of simulated stock returns over time period t
        
        simulate annual rate of return as sim return = mean + Z*sigma
        where Z = randomly drawn number from standard normal distribution
        
        simulated return(dt) = (mean - sigma**2 / 2)*dt + Z * sigma * sqr(dt)
        
        dt = 1 / nper_per_year
        """
        
        #creating variables for dt and time for range
        dt = 1 / self.nper_per_year
        time = int(self.nper_per_year * self.t)
        
        #creating empty list
        blank = np.empty((0, 0))
    

        #for loop to iterate through
        for x in range(time):
            
            #getting random Z value
            z = np.random.normal()
            
            sim = ((self.mu - (self.sigma**2 )/2) * dt ) + (z * self.sigma * math.sqrt(dt))
            
            #appending the value
            blank = np.append(blank, sim)
        
        
        return blank
            

    #Method 3
    def generate_simulated_stock_values(self):
        """This method generates and returns an
        np.array containing a sequence of stock values
        corresponding to a random sequences of stock 
        returns.
    
        Si = Si-1 * e^(ri-1)"""
        
        #creating an empty np array for the prices
        prices = np.empty((0, 0))
        
        #gemerating array of the returns
        returns = self.generate_simulated_stock_returns()
        
        #adding initial value to the list of prices
        current = self.s
        prices = np.append(prices, current)

        
        #for loop iterating through length of returns
        for x in range(len(returns)):
            
            #variables for current return and price
            current_r = returns[x]
            current_price = prices[x]
            
            #getting value of stock and adding to list of prices
            Si = current_price * (math.e**(current_r))
            prices = np.append(prices, Si)
        

        
        return prices
    
    
    #Method 4
    def plot_simulated_stock_values(self, num_trials = 1):
        """Method generates a plot of num_trails series of
        simulated stock returns. Num_trials = optional parameter.
        
        Uses matplotlib library for the plots"""
        
        
        #creating empty np array for x values
        x_val =  np.empty((0, 0))
        
        #getting variable for time
        time = int(self.nper_per_year * self.t)
        
        
        #iterating through adding time values to x_val table
        for x in range(time + 1):
            
            value = x / self.nper_per_year
            x_val = np.append(x_val, value)
        
        
        
        #for loop to get the values for each trial
        for x in range(num_trials):
            
            #getting the current generated values and plotting
            current = self.generate_simulated_stock_values() 
            plt.plot(x_val, current)
            

        #Creating titles and labels for graph
        plt.title(f'{num_trials} Simulated Trials')
        plt.xlabel('Years')
        plt.ylabel('Stock Value $')
        plt.show()





if __name__ == '__main__':
    sim = MCStockSimulator(100, .5, 0.1, 0.3, 6)
    
    print(sim)
    
    returns = sim.generate_simulated_stock_returns()
    
    print(returns)
    
    values = sim.generate_simulated_stock_values()
    
    print(values)
    
    plot = sim.plot_simulated_stock_values(5)
    
    print(plot)
    
