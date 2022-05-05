# PortfolioRebalancing


This set of files computes and compares different types of portfolios with and wihtout rebalancing.

The file 'PortConstructionRebalance.py' creates equally weighted porfolios and rebalances portfolios. 
There are additional files in the folder used to compute numbers for the main file: PortfolioComp.py
These include:

- MonteCarloSimulation.py (this file computes monte carlo simulation and normal distribution stock values)
- ModelVaR.py (this file compute model value at risk and historical VaR)
- MinVarPort_EfficientPort.py (this file computes global minimum variance portfolios, minimum variance portfolios, and stdev of efficient portfolio)
- DrawdownMCSim.py (this file computes and plots a portfolios drawdown, the file also will compute maximum drawdown using Monte Carlo Simulations

In the main file, PortfolioComp.py, it compares different portfolios made up of a number of historical stock values held in csv files
The file constructs:
- An equally weighted portfolio with rebalancing
- An equally weighted portfolio wihtout rebalancing
- A mean variance portfolio efficient portfolio acheiving same returns as equally weighted -- with rebalance
- A mean variance portfolio efficient portfolio acheiving same returns as equally weighted -- without rebalance

A look at the a13graphs.pdf shows the outcome using the following stocks:
JPM, GOOG, IBM, FB & GM
