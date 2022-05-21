import numpy as np
import time

def f(x):
    return np.power(x,3)

def Trapezoidal_Vectorize(start, end, N):
    eta = (end - start)/N
    k = [(start + j * eta) for j in np.arange(1, N)]
    f_k = f(k)
    sum_f_k = (sum(f_k) + 0.5*(f(start) + f(end))) * eta

    return sum_f_k


def Trapezoidal(start, end, N):

    eta = (end - start)/N
    somme = 0
    for j in np.arange(0,N+1):
        k = start + j*eta
        score = f(k)

        if (j == start) | (j == N):
            wJ = 1 / 2
        else:
            wJ = 1

        somme += eta *score * wJ

    return somme


start_time = time.time()
print(Trapezoidal(start = 1, end = 10, N=1000))
elapsed_time = time.time() - start_time
print('Evaluation using simple solution: ' + str(round(elapsed_time, 3)) + 'sec')

start_time = time.time()
print(Trapezoidal_Vectorize(start = 1, end = 10, N=1000))
elapsed_time = time.time() - start_time
print('Evaluation using vectorize solution: ' + str(round(elapsed_time, 3)) + 'sec')
