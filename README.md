# Options pricing
The aim of this project is to price many types of option under different models. 

# main.py: Binomial Models
In the main.py script, one can find how to evaluate an European and American options under the Binomial model.

# Portfolio metrics.py: Black Scholes and Monte Carlo Simulation
In the Portfolio metrics.py script, one can find the evaluation of European and Binary options under the BS formula.
The script, also, gives access to the pricing of Asian and different type of Barrier options which are done under monte carlo simulation (have to be completed).
A function which allows to build a portfolio of options and to observe the greeks of the portfolio(only availible for Binary and European so far) is also availible.

# Volatility_smile.py: Newton-Raphston algorithm applied to the construction of the volatility surface
In this script, one can find the construction of the volatility smile using Newton-Raphston algorithm. One can be inspired by this script to prove that the BS assumption of constant volatility does not hold on the real market.

# Pricing_European_Numerical_Integration.py

One can find, in this script, a method to price European option using numerical integration. The pricer assumes that prices are log normally distributed and the evaluation of the integral is done using Trapezoidal method. The lecture "Computational Methods in Pricing and Model Calibration" delivered by Columbia University has been the support of this project, and the technique they used is really close to this one. 

Note: I am aware that errors/ wrong implementations are possible and I please you to send me an email if you notice an (some) error(s). It would be very helpful, and will help me to improve myself. 
Here is my mail address: nicolas.manelli013@gmail.com
