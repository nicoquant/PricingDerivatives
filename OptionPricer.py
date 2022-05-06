import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats.mstats import gmean


def factorial(n):
    if n in [1,0]:
        return 1
    else:
        return n*factorial(n-1)

def combinaison(n, i):
    return factorial(n) / (factorial(n-i)*factorial(i))


def EuropeanOptionBinomial(S0, K, T, r,u , d, p, N, type_, output = None):

    # Probability of each paths at maturity
    weight = [combinaison(N, i) * p ** i * (1 - p) ** (N - i) for i in range(N + 1)]

    # Simulating stock price at maturity
    ST = S0 * (u) ** (np.arange(N + 1)) * (d) ** (N - np.arange(N + 1))

    # Payoff at maturity
    if type_ == 'c':
        C = np.maximum(ST - K, 0)
    elif type_ == 'p':
        C = np.maximum(K - ST, 0)
    else:
        raise ValueError("type_ must be 'c' or 'p'")

    if output == 'payoff':
        # useful only when pricing American Options
        return C
    return ((weight * C) * np.exp(-r * T)).sum()


def AmericanOptionBinomial(S0, K, T, r, u, d, p, N, type_):
    dt = T / N
    disc = np.exp(-r * dt)

    # Stock prices at maturity
    C = EuropeanOptionBinomial(S0, K, T, r, u, d, p, N, type_, output='payoff')

    # Backward process
    for i in np.arange(N - 1, -1, -1):
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))  # create path (vector)
        C[i] = disc * (p * C[i + 1] + (1 - p) * C[i])  # price of the computed period (vector)

        if type_ == 'p':
            C = np.maximum(K - S, C[i])  # output: maximum between price of the period and K(int) - S
        elif type_ == 'c':
            C = np.maximum(S - K, C[i])
# ALTERNATIVE LOOP
#    for i in np.arange(N - 1, -1, -1):
#        for j in range(0, i + 1):
#            S = S0 * u ** j * d ** (i - j)
#            C[j] = disc * (p * C[j + 1] + (1 - p) * C[j])
#              if type_ == 'p':
#                C[j] = max(C[j], K - S)
#            else:
#                C[j] = max(C[j], S - K)
    return C[0]


class European_BS:
    def __init__(self, q, r, vol, T):

        self.q = q
        self.r = r
        self.vol = vol
        self.T = T

    @staticmethod
    def N_(x):
        '''

        :param x: number
        :return:
        '''
        return 1 / np.sqrt(2 * np.pi) * np.exp(-x / 2)

    def delta(self, short_long, type, S, K, T):
        """
        :param self.q: dividends
        :param T: maturity
        :param d1: comes from BS
        :param type: put or call
        :return:
        """
        if short_long == 'short':
            l = -1
        elif short_long == 'long':
            l = 1
        if type == 'call':
            return np.exp(-self.q * T) * norm.cdf(self.d1_BS(S, K, T)) * l
        elif type == 'put':
            return np.exp(-self.q * T) * (norm.cdf(self.d1_BS(S, K, T)) - 1) * l

    def gamma(self, short_long, S, K, T, type = None):
        '''

        :param S: StocK Price
        :param self.q: Dividends
        :param T: Maturity
        :param d1: From BS
        :param self.vol: standard deviation
        :return: Gamma
        '''
        if short_long == 'short':
            l = -1
        elif short_long == 'long':
            l = 1
        return self.N_(self.d1_BS(S, K, T)) * np.exp(-self.q * T) / (S * self.vol * np.sqrt(T)) * l

    def vega(self, short_long, S, K, T,type = None):
        if short_long == 'short':
            l = -1
        elif short_long == 'long':
            l = 1
        return S * np.exp(-self.q * T) * self.N_(self.d1_BS(S, K, T)) * np.sqrt(T) * l

    def theta(self, short_long, type, S, K, T):
        '''
        :param d2: from BS
        :param r: rate
        :param self.vol: standard deviation
        :return:
        '''
        if short_long == 'short':
            l = -1
        elif short_long == 'long':
            l = 1
        if type == 'call':
            return (-(S * self.vol * np.exp(-self.q * T) * self.N_(self.d1_BS(S, K, T)) / (
                    2 * np.sqrt(T))) - self.r * K * np.exp(-self.r * T) * norm.cdf(
                self.d2_BS(S, K, T)) + self.q * S * np.exp(-self.q * T) * norm.cdf(self.d1_BS(S, K, T))) * l

        elif type == 'put':
            return (-(S * self.vol * np.exp(-self.q * T) * self.N_(self.d1_BS(S, K, T)) / (
                    2 * np.sqrt(T))) + self.r * K * np.exp(-self.r * T) * norm.cdf(
                self.d2_BS()) - self.q * S * np.exp(-self.q * T) * norm.cdf(self.d1_BS(S, K, T))) * l

    def call_payoff(self, n, S, K):
        '''

        :param n: number of calls bought
        :param S: Price of the underlying at maturity
        :param K: StriKe
        :return: payoff call
        '''
        return n * max([0, S - K])

    def put_payoff(self, n, S, K):
        '''

        :param n: number of calls bought
        :param S: Price of the underlying at maturity
        :param K: StriKe
        :return: payoff call
        '''
        return n * max([0, K - S])

    def forward(self, S, K, T):
        return S * np.exp(self.q * T) - K * np.exp(-self.r * T)

    def call_european(self, S, K, T):
        return S * np.exp(-self.q * T) * norm.cdf(self.d1_BS(S, K, T)) - K * np.exp(
            -self.r * T) * norm.cdf(self.d2_BS(S, K, T))

    def put_european(self, S, K, T):
        return S * np.exp(-self.q * T) * (norm.cdf(self.d1_BS()) - 1) - K * np.exp(-self.r * T) * (
                norm.cdf(self.d2_BS(S, K)) - 1)

    def d1_BS(self, S, K, T):
        return (np.log(S / K) + (self.r - self.q + (self.vol ** 2) / 2) * T) / (self.vol * T)

    def d2_BS(self, S, K, T):
        return self.d1_BS(S, K, T) - self.vol * np.sqrt(T)


class BinaryOption(European_BS):
    def __init__(self, q, r, vol, T):
        super().__init__(q, r, vol, T)

    def call_binary(self, short_long, S, K, T, x):
        if short_long == 'long':
            return np.exp(-(self.r - self.q) * T) * norm.cdf(x*European_BS.d2_BS(self,S, K, T))
        elif short_long == 'short':
            return -np.exp(-(self.r - self.q) * T) * norm.cdf(x * European_BS.d2_BS(self, S, K, T))
    def put_binary(self,short_long,S, K, T, x):
        if short_long == 'long':
            return np.exp(-(self.r - self.q) * T) * norm.cdf(1-x*European_BS.d2_BS(self,S, K, T))
        if short_long == 'put':
            return -np.exp(-(self.r - self.q) * T) * norm.cdf(1 - x * European_BS.d2_BS(self, S, K, T))

    def delta_binary(self, short_long,type, S, K, T, x):
        if short_long == 'short':
            l = -1
        elif short_long == 'long':
            l = 1
        if type == 'call':
            return x * np.exp(-(self.r - self.q) * T) * European_BS.N_(European_BS.d2_BS(self, K, S, T)) / (self.vol * S * np.sqrt(T)) * l
        elif type == 'put':
            return -x * np.exp(-(self.r - self.q) * T) * European_BS.N_(European_BS.d2_BS(self, K, S, T)) / (self.vol * S * np.sqrt(T)) * l

    def gamma_binary(self,short_long,type,  S, K, T, x = None):
        '''
        Essentially 0 close to expiration. Change signs when Strike passes through K
        :return:
        '''
        if short_long == 'short':
            l = -1
        elif short_long == 'long':
            l = 1
        if type == 'call':
            return -np.exp(-(self.r - self.q) * T) * European_BS.d1_BS(self, S, K, T) * norm.cdf(European_BS.d2_BS(self,S,K,T)) / (S**2 * self.vol **2 * T)*l
        elif type == 'put':
            return np.exp(-(self.r - self.q) * T) * European_BS.d1_BS(self, S, K, T) * norm.cdf(European_BS.d2_BS(self,S,K,T)) / (S**2 * self.vol **2 * T)*l

    def theta_binary(self,short_long,type,S, K, T, x = None):
        if short_long == 'short':
            l = -1
        elif short_long == 'long':
            l = 1
        if type == 'call':
            return l *(self.r * np.exp(-(self.r - self.q) * T) * norm.cdf(European_BS.d2_BS(self,S,K,T)) + np.exp(-(self.r - self.q) * T) * European_BS.N_(European_BS.d2_BS(self,S,K,T)) * (self.d1_BS(S,K,T)/(2*T) - (self.r - self.q)/(self.vol*np.sqrt(T))))
        elif type =='put':
            return l*(self.r * np.exp(-(self.r - self.q) * T) * norm.cdf(European_BS.d2_BS(self,S,K,T)) - np.exp(-(self.r - self.q) * T) * European_BS.N_(European_BS.d2_BS(self,S,K,T)) * (self.d1_BS(S,K,T)/(2*T) - (self.r - self.q)/(self.vol*np.sqrt(T))))


    def vega_binary(self,short_long,type, S, K, T, x = None):
        '''
        Not relevant for binary: Since the gamma can be positive or negative.
        Binary are of course exposed to vol but it's not computed using vega (chap 52 Wilmott)
        :return:
        '''
        if short_long == 'short':
            l = -1
        elif short_long == 'long':
            l = 1
        if type == 'call':
            return (-np.exp(-(self.r - self.q) * T) * European_BS.N_(European_BS.d2_BS(self,S, K, T)) * European_BS.d1_BS(self,S, K, T) / self.vol)*l
        elif type == 'put':
            return (np.exp(-(self.r - self.q) * T) * European_BS.N_(European_BS.d2_BS(self,S, K, T)) * European_BS.d1_BS(self,S, K,
                                                                                                       T) / self.vol)*l


class AsianOptionMCM(European_BS):
    def __init__(self, q, r, vol, T):
        super().__init__(q, r, vol, T)
    @staticmethod
    def geo_mean(iterable):
        a = np.array(iterable)
        return gmean(a)#a.prod() ** (1.0 / len(a))

    def GBM(self, S, steps, n_paths, T):
        Z = np.random.normal(0.0, 1.0, [n_paths, n_paths])
        W = np.zeros([n_paths, n_paths + 1])

        dt = T/steps

        S1 = np.zeros([n_paths, steps + 1])
        S1[:,0] = S

        for i in range(0, steps):
            W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
            S1[:,i + 1] = S1[:,i] + S1[:,i] * self.r * dt + S1[:,i]*self.vol*(W[:, i + 1] - W[:, i])

        return S1

    def call_asian(self, S, K, steps, n_paths, T):
        '''
        Compute 10 000 paths using GBM
        Compute the payoff of the last values of the paths
        Transform neg values in 0
        compute mean of those payoffs
        :return:
        '''
        matrice = self.GBM(S, steps, n_paths, T)
        payoff = matrice[:, -1] - K
        payoff[payoff < 0] = 0
        if len(payoff) > 0:
            return np.mean(payoff)*np.exp(-self.r*T)
        else:
            return 0
    def put_asian(self, S, K, steps, n_paths, T):
        matrice = self.GBM(S, steps, n_paths, T)
        payoff = matrice[:, -1] - K
        payoff = payoff*(-1)
        payoff[payoff < 0] = 0
        if len(payoff) > 0:
            return np.mean(payoff)*np.exp(-self.r*T)
        else:
            return 0



class portfolio(European_BS):
    def __init__(self,q, r, vol, T):
        super().__init__(q, r, vol, T)
        self.options_ptf = {}

    def add(self,option_type, n, long_put, type, Stock_Price, Strike, T):
        for i in range(0, n):
            self.options_ptf[str(len(self.options_ptf) + 1)] = [option_type,long_put, type, Stock_Price, Strike, T]


    def compute_greeks(self):
        d = sum([self.delta(self.options_ptf.get(f'{i}')[1], self.options_ptf.get(f'{i}')[2],
                            self.options_ptf.get(f'{i}')[3], self.options_ptf.get(f'{i}')[4],
                            self.options_ptf.get(f'{i}')[5]) for i in range(1, len(self.options_ptf) + 1)])
        g = sum([self.gamma(self.options_ptf.get(f'{i}')[1], self.options_ptf.get(f'{i}')[3],
                            self.options_ptf.get(f'{i}')[4], self.options_ptf.get(f'{i}')[5]) for i in
                 range(1, len(self.options_ptf) + 1)])
        v = sum([self.vega(self.options_ptf.get(f'{i}')[1], self.options_ptf.get(f'{i}')[3],
                           self.options_ptf.get(f'{i}')[4], self.options_ptf.get(f'{i}')[5]) for i in
                 range(1, len(self.options_ptf) + 1)])
        t = sum([self.theta(self.options_ptf.get(f'{i}')[1], self.options_ptf.get(f'{i}')[2],
                            self.options_ptf.get(f'{i}')[3],
                            self.options_ptf.get(f'{i}')[4], self.options_ptf.get(f'{i}')[5]) for i in
                 range(1, len(self.options_ptf) + 1)])
        return {'delta': d, 'gamma': g, 'vega': v, 'theta': t}

    def compute_greeks2(self, greek):
        greek_list = []
        for i in wallet.keys():
            if wallet.get(f'{i}')[0] == 'euro':
                #greek = getattr(European_BS, greek)
                greek_list.append(getattr(globals()['met'], greek)(short_long = wallet.get(f'{i}')[1], type = wallet.get(f'{i}')[2],
                                       S = wallet.get(f'{i}')[3], K = wallet.get(f'{i}')[4],
                                       T = wallet.get(f'{i}')[5]))
            elif wallet.get(f'{i}')[0] == 'binary':
                greek = greek+'_binary'
                greek_list.append(getattr(globals()['bin'], greek)(short_long=wallet.get(f'{i}')[1], type=wallet.get(f'{i}')[2],
                                                     S=wallet.get(f'{i}')[3], K=wallet.get(f'{i}')[4],
                                                     T=wallet.get(f'{i}')[5], x = 1))
        return sum(greek_list)

