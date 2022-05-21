import math
import cmath
import numpy as np
from OptionPricer import *
import time


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

def genericCF(v, S0, r, q, T, vol):
    '''

    :param v: Point for which the CF is evaluated
    :param S0: Stock price T = 0
    :param r: rate
    :param q: dividend
    :param T: maturity
    :param vol: sqrt(volatility)
    :return: Characteristic Function of the log of the stock price
    '''
    return np.exp(1j*(np.log(S0)+(r-q-vol**2/2)*T)*v - (vol**2 * v**2)*T/2)


def Pricing_Numerical_Integration_Fourier_Transform(S, K, r, q, T, N, vol, alpha, eta):
    k = np.log(K)
    df = np.exp(-r * T) # discount factor
    sum_tot = 0


    for j in range(N):
        nuJ = j * eta

        # Fourier Transform of modified call: ct(k) = exp(alpha*k) * Ct(k)
        psy_v = df * genericCF(nuJ - (alpha + 1) * 1j, S, r, q, T, vol) / ((alpha + 1j * nuJ)*(alpha + 1j * nuJ + 1))
        if j == 0:
            w = eta/2
        else:
            w = eta

        sum_tot += np.exp(-1j * nuJ * k) * psy_v * w # Inverse Fourier Transform to find modified call

    Ct_k = (np.exp(-alpha * k)/math.pi) * sum_tot # Ct(k) = exp(-alpha*k) * ct(k)

    return np.real(Ct_k)

def Pricing_Numerical_Integration_Fourier_Transform_vectorize(S, K, r, q, T, N, vol, alpha, eta):
    '''
    :return: Same methods as below but vectorize to speed the
    '''
    k = np.log(K)
    df = np.exp(-r * T) # discount factor
    sum_tot = np.exp(-1j * 0 * k) * df * genericCF(0 - (alpha + 1) * 1j, S, r, q, T, vol)*eta / (2*(alpha + 1j * 0)*(alpha + 1j * 0 + 1))

    nuJ = np.array([(j * eta) for j in np.arange(1,N)])

    sum_tot += sum(np.exp(-1j * nuJ * k) * eta * df * genericCF(nuJ - (alpha + 1) * 1j, S, r, q, T, vol) / ((alpha + 1j * nuJ)*(alpha + 1j * nuJ + 1)))

    Ct_k = (np.exp(-alpha * k)/math.pi) * sum_tot # Ct(k) = exp(-alpha*k) * ct(k)

    return np.real(Ct_k)



if __name__ == '__main__':
    S0, K, r, q, vol, T = 100, 80, 0.05, 0.01, 0.3, 1.0
    eta = 0.25  # step-size

    # number of grid points
    n = 12
    N = 2 ** n

    arg = (r, q, S0, K, vol, T, N, eta)
    start_time = time.time()
    c0_KT, p0_KT = PricingNumericalIntegration(*arg) # c0_KT = 25.61
    elapsed_time = time.time() - start_time
    print('Pricing took ' + str(round(elapsed_time,3)) + ' seconds')

    Euro = European_BS(q=q, r=r, vol=vol, T=T)
    BS_Call_Price = Euro.call_european(S0, K, T)  # 25.61

    start_time = time.time()
    C_FT = Pricing_Numerical_Integration_Fourier_Transform(S0, K, r, q, T, N, vol, 1.5, eta) # 25.61
    elapsed_time = time.time() - start_time
    print('Pricing using FT took ' + str(round(elapsed_time,3)) + ' seconds')

    start_time = time.time()
    C_FT_vect = Pricing_Numerical_Integration_Fourier_Transform_vectorize(S0, K, r, q, T, N, vol, 1.5, eta) # 25.61
    elapsed_time = time.time() - start_time
    print('Pricing using FT vectorized took ' + str(round(elapsed_time,3)) + ' seconds') 
