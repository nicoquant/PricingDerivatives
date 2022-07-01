import numpy as np
import pandas as pd
from scipy.stats import norm
import pandas_datareader as pdr
import matplotlib.pyplot as plt

def call_european(S, K, T, q, r, vol):
    '''

    :params: BS parameters
    :param vol: sigma that we guess
    :return: price of an european call
    '''
    return S * np.exp(-q * T) * norm.cdf(d1_BS(S, K, T, r,q, vol)) - K * np.exp(
        -r * T) * norm.cdf(d2_BS(S, K, T, r, q, vol))

def d1_BS(S, K, T, r, q, vol):
    '''

    :params: BS parameters
    :return: d1
    '''
    return (np.log(S / K) + (r - q + (vol ** 2) / 2) * T) / (vol * T)

def d2_BS(S, K, T, r, q, vol):
    '''

    :params: BS parameters
    :return: d2
    '''
    return d1_BS(S, K, T, r, q, vol) - vol * np.sqrt(T)

def ImpliedVol(S, K, T,q,r,sigma_guess, mkt_price, max_try = 200, tolerence = 0.001):
    '''
    Newton Raphston Algorithm used to find Implied volatility

    :params of the BS
    :param sigma_guess: guess a sigma
    :param mkt_price: market price of the call
    :return: Implied vol of the call
    '''
    def function(sigma_guess):
        '''

        :param sigma_guess: sigma old
        :return: absolute value of price difference betewwen market price and BS(sigma old)
        '''
        return np.abs(call_european(S, K, T, q, r, vol = sigma_guess) - mkt_price)

    for i in range(max_try + 1):
        priceBS = call_european(S, K, T, q, r, vol = sigma_guess)
        vega = (S * np.exp(-q * T) * norm.pdf(d1_BS(S, K, T, r, q, sigma_guess)) * np.sqrt(T))

        sigma_new = sigma_guess - (priceBS - mkt_price)/vega

        if function(sigma_new) < tolerence:
            break
        sigma_guess = sigma_new
    return sigma_new

a = ImpliedVol(S = 100, K = 102,T = 164/365,q = 0, r = 0.2, sigma_guess=0.20, mkt_price=22)

import yfinance as yf
from datetime import datetime
date_exp = '2022-05-27'
ticker = 'AMZN'
stock= yf.Ticker(ticker)
opt = stock.option_chain(date=date_exp)

calls = opt.calls
calls['Expiration'] = datetime.strptime(date_exp, "%Y-%m-%d")
calls['Time'] = 0.7
calls['Stock'] = yf.download(ticker,
                      start='2021-12-01',
                      end='2022-10-05',
                      progress=False,)['Adj Close'][-1]
ivs = []
for row in calls.itertuples():
    ivs.append(ImpliedVol(row.Stock, row.strike, row.Time, 0, 0.02, 0.3, row.lastPrice))


plt.scatter(calls.strike, ivs, label='calculated')
plt.xlabel('Strike')
plt.ylabel('Implied Vol')
plt.title('Implied Volatility Curve')
plt.legend()
plt.show()
