import numpy as np
from OptionPricer import *


def Log_Normal(S, r, q, vol, S0, T):
    f = np.exp(-0.5 * ((np.log(S / S0) - (r - q - vol ** 2 / 2) * T) / (vol * np.sqrt(T))) ** 2) / (
            vol * S * np.sqrt(2 * np.pi * T))
    return f


def PricingNumericalIntegration(*args):
    r, q, S0, K, vol, T, N, dS = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]

    # discount factor
    df = np.exp(-r * T)

    ####### Evaluation of the integral using Trapezoidal method #######
    S = np.arange(1, N * dS + 1, dS)  # Define Grid of Prices

    tmp = Log_Normal(S, r, q, vol, S0, T)  # log normal densities of S0

    w = [dS] * N
    w[0] = dS / 2

    Call = 0
    Put = 0
    for j in range(N):
        if (S[j] > K):
            Call += (S[j] - K) * tmp[j] * w[j]  # addition of sub intervals

        if (S[j] < K):
            Put += (K - S[j]) * tmp[j] * w[j]  # addition of sub intervals

    Call_Price = df * Call
    Put_Price = df * Put

    return Call_Price, Put_Price


if __name__ == '__main__':
    S0, K, r, q, vol, T = 100, 80, 0.05, 0.01, 0.3, 2.0
    dS = 0.10  # step-size

    # number of grid points
    n = 12
    N = 2 ** n

    arg = (r, q, S0, K, vol, T, N, dS)
    c0_KT, p0_KT = PricingNumericalIntegration(*arg)

    Euro = European_BS(q=q, r=r, vol=vol, T=T)  # 30.41
    BS_Call_Price = Euro.call_european(S0, K, T)  # 30.11
