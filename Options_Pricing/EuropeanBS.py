import numpy as np
from scipy.stats import norm


class European_BS:
    def __init__(self, q, r, vol, T):

        self.q = q
        self.r = r
        self.vol = vol
        self.T = T

    @staticmethod
    def N_(x):
        """

        :param x: number
        :return:
        """
        return 1 / np.sqrt(2 * np.pi) * np.exp(-x / 2)

    def delta(self, short_long, type, S, K, T):
        """
        :param self.q: dividends
        :param T: maturity
        :param d1: comes from BS
        :param type: put or call
        :return:
        """
        if short_long == "short":
            l = -1
        elif short_long == "long":
            l = 1
        if type == "call":
            return np.exp(-self.q * T) * norm.cdf(self.d1_BS(S, K, T)) * l
        elif type == "put":
            return np.exp(-self.q * T) * (norm.cdf(self.d1_BS(S, K, T)) - 1) * l

    def gamma(self, short_long, S, K, T, type=None):
        """

        :param S: StocK Price
        :param self.q: Dividends
        :param T: Maturity
        :param d1: From BS
        :param self.vol: standard deviation
        :return: Gamma
        """
        if short_long == "short":
            l = -1
        elif short_long == "long":
            l = 1
        return (
            self.N_(self.d1_BS(S, K, T))
            * np.exp(-self.q * T)
            / (S * self.vol * np.sqrt(T))
            * l
        )

    def vega(self, short_long, S, K, T, type=None):
        if short_long == "short":
            l = -1
        elif short_long == "long":
            l = 1
        return S * np.exp(-self.q * T) * self.N_(self.d1_BS(S, K, T)) * np.sqrt(T) * l

    def theta(self, short_long, type, S, K, T):
        """
        :param d2: from BS
        :param r: rate
        :param self.vol: standard deviation
        :return:
        """
        if short_long == "short":
            l = -1
        elif short_long == "long":
            l = 1
        if type == "call":
            return (
                -(
                    S
                    * self.vol
                    * np.exp(-self.q * T)
                    * self.N_(self.d1_BS(S, K, T))
                    / (2 * np.sqrt(T))
                )
                - self.r * K * np.exp(-self.r * T) * norm.cdf(self.d2_BS(S, K, T))
                + self.q * S * np.exp(-self.q * T) * norm.cdf(self.d1_BS(S, K, T))
            ) * l

        elif type == "put":
            return (
                -(
                    S
                    * self.vol
                    * np.exp(-self.q * T)
                    * self.N_(self.d1_BS(S, K, T))
                    / (2 * np.sqrt(T))
                )
                + self.r * K * np.exp(-self.r * T) * norm.cdf(self.d2_BS())
                - self.q * S * np.exp(-self.q * T) * norm.cdf(self.d1_BS(S, K, T))
            ) * l

    def call_payoff(self, n, S, K):
        """

        :param n: number of calls bought
        :param S: Price of the underlying at maturity
        :param K: StriKe
        :return: payoff call
        """
        return n * max([0, S - K])

    def put_payoff(self, n, S, K):
        """

        :param n: number of calls bought
        :param S: Price of the underlying at maturity
        :param K: StriKe
        :return: payoff call
        """
        return n * max([0, K - S])

    def forward(self, S, K, T):
        return S * np.exp(self.q * T) - K * np.exp(-self.r * T)

    def call_european(self, S, K, T):
        return S * np.exp(-self.q * T) * norm.cdf(self.d1_BS(S, K, T)) - K * np.exp(
            -self.r * T
        ) * norm.cdf(self.d2_BS(S, K, T))

    def put_european(self, S, K, T):
        return S * np.exp(-self.q * T) * (
            norm.cdf(self.d1_BS(S, K, T)) - 1
        ) - K * np.exp(-self.r * T) * (norm.cdf(self.d2_BS(S, K, T)) - 1)

    def d1_BS(self, S, K, T):
        return (np.log(S / K) + (self.r - self.q + (self.vol ** 2) / 2) * T) / (
            self.vol * T
        )

    def d2_BS(self, S, K, T):
        return self.d1_BS(S, K, T) - self.vol * np.sqrt(T)
