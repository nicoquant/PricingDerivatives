import numpy as np

def factorial(n):
    if n in [1,0]:
        return 1
    else:
        return n*factorial(n-1)

def combinaison(n, i):
    return factorial(n) / (factorial(n-i)*factorial(i))


def EuropeanOption(S0, K, T, r,u , d, p, N, type_, output = None):

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
    return (weight * C) * np.exp(-r * T)


def AmericanOption(S0, K, T, r, u, d, p, N, type_):
    dt = T / N
    disc = np.exp(-r * dt)

    # Stock prices at maturity
    C = EuropeanOption(S0, K, T, r, u, d, p, N, type_, output='payoff')

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