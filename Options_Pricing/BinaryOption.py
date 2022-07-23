from EuropeanBS import *

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
        if short_long == 'short':
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