from EuropeanBS import *
import numpy as np
from scipy.stats import norm
from scipy.stats.mstats import gmean


class OptionMCM(European_BS):
    def __init__(self, q, r, vol, T):
        super().__init__(q, r, vol, T)

    @staticmethod
    def geo_mean(iterable):
        a = np.array(iterable)
        return gmean(a)  # a.prod() ** (1.0 / len(a))

    def GBM(self, S, steps, n_paths, T):
        Z = np.random.normal(0.0, 1.0, [n_paths, n_paths])
        W = np.zeros([n_paths, n_paths + 1])

        dt = T / steps

        S1 = np.zeros([n_paths, steps + 1])
        S1[:, 0] = S

        for i in range(0, steps):
            W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
            S1[:, i + 1] = (
                S1[:, i]
                + S1[:, i] * self.r * dt
                + S1[:, i] * self.vol * (W[:, i + 1] - W[:, i])
            )

        return S1

    def asian_mcm(self, S, K, steps, n_paths, T, type):
        """
        Compute 10 000 paths using GBM
        Compute the payoff of the last values of the paths
        Transform neg values in 0
        compute mean of those payoffs
        :return:
        """
        matrice = self.GBM(S, steps, n_paths, T)
        if type == "p":
            payoff = np.maximum(K - matrice[:, -1][matrice[:, -1] != 0], 0)

        elif type == "c":
            payoff = np.maximum(matrice[:, -1][matrice[:, -1] != 0] - K, 0)

        if len(payoff) > 0:
            return np.sum(payoff) * np.exp(-self.r * T) / n_paths
        else:
            return 0

    # Barrier Options: Not efficiently coded
    def put_up_out(self, S, K, H, steps, n_paths, T):

        matrice = self.GBM(S, steps, n_paths, T)
        simulation_in = matrice.copy()

        condition = np.unique(np.where((matrice >= H))[0])

        simulation_in[condition, :] = 0

        payoff = np.maximum(K - simulation_in[:, -1][simulation_in[:, -1] != 0], 0)
        return np.exp(-self.r * self.T) * np.sum(payoff) / n_paths

    def put_up_in(self, S, K, H, steps, n_paths, T):

        if H < S:
            raise ValueError("H has to be greater than S, for a PUI")
        matrice = self.GBM(S, steps, n_paths, T)
        simulation_in = matrice.copy()

        condition = np.unique(np.where((matrice >= H))[0])

        simulation_in[~condition, :] = 0

        payoff = np.maximum(K - simulation_in[:, -1][simulation_in[:, -1] != 0], 0)
        return np.exp(-self.r * self.T) * np.sum(payoff) / n_paths

    def put_down_in(self, S, K, H, steps, n_paths, T):

        matrice = self.GBM(S, steps, n_paths, T)
        simulation_in = matrice.copy()

        condition = np.unique(np.where((matrice <= H))[0])

        simulation_in[~condition, :] = 0

        payoff = np.maximum(K - simulation_in[:, -1][simulation_in[:, -1] != 0], 0)
        return np.exp(-self.r * self.T) * np.sum(payoff) / n_paths

    def put_down_out(self, S, K, H, steps, n_paths, T):

        matrice = self.GBM(S, steps, n_paths, T)
        simulation_in = matrice.copy()

        condition = np.unique(np.where((matrice <= H))[0])

        simulation_in[condition, :] = 0

        payoff = np.maximum(K - simulation_in[:, -1][simulation_in[:, -1] != 0], 0)
        return np.exp(-self.r * self.T) * np.sum(payoff) / n_paths

    def call_down_out(self, S, K, H, steps, n_paths, T):

        matrice = self.GBM(S, steps, n_paths, T)
        simulation_in = matrice.copy()

        condition = np.unique(np.where((matrice <= H))[0])

        simulation_in[condition, :] = 0

        payoff = np.maximum(K - simulation_in[:, -1][simulation_in[:, -1] != 0], 0)
        return np.exp(-self.r * self.T) * np.sum(payoff) / n_paths

    def call_down_in(self, S, K, H, steps, n_paths, T):

        matrice = self.GBM(S, steps, n_paths, T)
        simulation_in = matrice.copy()

        condition = np.unique(np.where((matrice <= H))[0])

        simulation_in[~condition, :] = 0

        payoff = np.maximum(K - simulation_in[:, -1][simulation_in[:, -1] != 0], 0)
        return np.exp(-self.r * self.T) * np.sum(payoff) / n_paths

    def call_up_out(self, S, K, H, steps, n_paths, T):

        matrice = self.GBM(S, steps, n_paths, T)
        simulation_in = matrice.copy()

        condition = np.unique(np.where((matrice >= H))[0])

        simulation_in[condition, :] = 0

        payoff = np.maximum(K - simulation_in[:, -1][simulation_in[:, -1] != 0], 0)
        return np.exp(-self.r * self.T) * np.sum(payoff) / n_paths

    def call_up_in(self, S, K, H, steps, n_paths, T):

        matrice = self.GBM(S, steps, n_paths, T)
        simulation_in = matrice.copy()

        condition = np.unique(np.where((matrice >= H))[0])

        simulation_in[~condition, :] = 0

        payoff = np.maximum(K - simulation_in[:, -1][simulation_in[:, -1] != 0], 0)
        return np.exp(-self.r * self.T) * np.sum(payoff) / n_paths
