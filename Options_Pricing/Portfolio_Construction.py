import numpy as np
from EuropeanBS import *


class portfolio(European_BS):
    def __init__(self, q, r, vol, T):
        super().__init__(q, r, vol, T)
        self.options_ptf = {}

    def add(self, option_type, n, long_put, type, Stock_Price, Strike, T):
        for i in range(0, n):
            self.options_ptf[str(len(self.options_ptf) + 1)] = [
                option_type,
                long_put,
                type,
                Stock_Price,
                Strike,
                T,
            ]

    def compute_greeks(self):
        d = sum(
            [
                self.delta(
                    self.options_ptf.get(f"{i}")[1],
                    self.options_ptf.get(f"{i}")[2],
                    self.options_ptf.get(f"{i}")[3],
                    self.options_ptf.get(f"{i}")[4],
                    self.options_ptf.get(f"{i}")[5],
                )
                for i in range(1, len(self.options_ptf) + 1)
            ]
        )
        g = sum(
            [
                self.gamma(
                    self.options_ptf.get(f"{i}")[1],
                    self.options_ptf.get(f"{i}")[3],
                    self.options_ptf.get(f"{i}")[4],
                    self.options_ptf.get(f"{i}")[5],
                )
                for i in range(1, len(self.options_ptf) + 1)
            ]
        )
        v = sum(
            [
                self.vega(
                    self.options_ptf.get(f"{i}")[1],
                    self.options_ptf.get(f"{i}")[3],
                    self.options_ptf.get(f"{i}")[4],
                    self.options_ptf.get(f"{i}")[5],
                )
                for i in range(1, len(self.options_ptf) + 1)
            ]
        )
        t = sum(
            [
                self.theta(
                    self.options_ptf.get(f"{i}")[1],
                    self.options_ptf.get(f"{i}")[2],
                    self.options_ptf.get(f"{i}")[3],
                    self.options_ptf.get(f"{i}")[4],
                    self.options_ptf.get(f"{i}")[5],
                )
                for i in range(1, len(self.options_ptf) + 1)
            ]
        )
        return {"delta": d, "gamma": g, "vega": v, "theta": t}

    def compute_greeks2(self, greek):
        wallet = self.options_ptf
        greek_list = []
        for i in wallet.keys():
            if wallet.get(f"{i}")[0] == "euro":
                # greek = getattr(European_BS, greek)
                greek_list.append(
                    getattr(globals()["met"], greek)(
                        short_long=wallet.get(f"{i}")[1],
                        type=wallet.get(f"{i}")[2],
                        S=wallet.get(f"{i}")[3],
                        K=wallet.get(f"{i}")[4],
                        T=wallet.get(f"{i}")[5],
                    )
                )
            elif wallet.get(f"{i}")[0] == "binary":
                greek = greek + "_binary"
                greek_list.append(
                    getattr(globals()["bin"], greek)(
                        short_long=wallet.get(f"{i}")[1],
                        type=wallet.get(f"{i}")[2],
                        S=wallet.get(f"{i}")[3],
                        K=wallet.get(f"{i}")[4],
                        T=wallet.get(f"{i}")[5],
                        x=1,
                    )
                )
        return sum(greek_list)
