# Options pricing
The aim of this project is to price many types of option under different models. 

One can find the evaluation of European and Binary options under the BS formula and Binomial model.
The script, also, gives access to the pricing of Asian and different type of Barrier options which are done under monte carlo simulation (have to be completed).
A function which allows to build a portfolio of options and to observe the greeks of the portfolio(only availible for Binary and European so far) is also availible.

# Volatility_smile.py: Newton-Raphston algorithm applied to the construction of the volatility surface
In this script, one can find the construction of the volatility smile using Newton-Raphston algorithm. One can be inspired by this script to prove that the BS assumption of constant volatility does not hold on the real market.

# Pricing_European_Numerical_Integration.py: Pricing by evaluating integrals

One can find, in this script, many methods to price European option using numerical integration. 
The pricer named "PricingNumericalIntegration" assumes that prices are log normally distributed and the evaluation of the integral is done using Trapezoidal method. First, the payoff curve of the option is fitted with the log normal distribution, then each intervals is evaluated. 

The pricer named "Pricing_Numerical_Integration_Fourier_Transform" starts using the characteritic function of BMS. Then, we compute the fourier transform of the modified call, evaluate the integral, use the Inverse Fourier Transform to find the modified call price and then transform the modified call price in the call price. 
The pricer "Pricing_Numerical_Integration_Fourier_Transform_vectorized" is the same as below but is vectorized but run faster (between 4 and 5 times faster).  

# Trapezoidal_Method.py
One can find in this script, 2 methods (one simple, one vectorized) to evaluate an integrale using the Trapezoidal Method.

# Autoregressive Models.py

One can find code (built from scratch) to fit the data to a model from the ARMA familly. 3 methods have been implemented to fit the data with an AR(p) (Yule Walker, Least Square and Conditional Maximum Likelihood), 1 method have been implemented to fit the data with the MA(q) and 1 method have been implemented to fit the data with the ARMA(p,q). Results are very close from those found using the statsmodels package.
I'll soon add a GARCH model.

Note: I am aware that errors/ wrong implementations are possible and I please you to send me an email if you notice an (some) error(s). It would be very helpful, and will help me to improve myself. 
Here is my mail address: nicolas.manelli013@gmail.com
