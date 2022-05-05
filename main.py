import numpy as np
from OptionPricer import EuropeanOptionBinomial, AmericanOptionBinomial

N = 2
S0 = 100
T = 2
sigma = np.log(1.1)
K = 100
r = 0.05
dt = T / N

u = round(np.exp(sigma * np.sqrt(dt)),1) # u = 1.1
d = round(np.exp(-sigma * np.sqrt(dt)),1) # d = 0.9

# probabilité up & down
p = (np.exp(r * dt) - d) / (u - d) # p = 0.7564


if __name__ == "__main__":
    Call_Euro = EuropeanOptionBinomial(S0, K, T, r, u, d, p, N, type_='c')
    Put_American = AmericanOptionBinomial(S0, K, T, r, u, d, p, N, type_='p')