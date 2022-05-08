import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats.mstats import gmean
from OptionPricer import *

if __name__ == '__main__':
    euro = European_BS(q=0.07, r=0.01, vol=0.2, T=1)
    bin = BinaryOption(q=0.07, r=0.01, vol=0.1, T=1)
    mcm = OptionMCM(q=0.07, r=0.01, vol=0.2, T=1)

    price_call = euro.call_european(100, 110, 2)
    delta_euro = euro.delta('long', 'call', 100, 110, 1)
    gamma_euro = euro.gamma('long', 100, 110, 1)
    vega_euro = euro.vega('long', 100, 110, 1)
    theta_euro = euro.theta('long', 'call', 100, 110, 1)

    ptf = portfolio(q=0.07, r=0.01, vol=0.1, T=1)
    wallet = ptf.options_ptf
    ptf.add('euro',1, 'long', 'call', 100, 105, 1)
    ptf.add('euro',3, 'long', 'call', 100, 112, 2)
    ptf.add('euro',1, 'short', 'call', 95, 100, 1)

    greeks_ptf = ptf.compute_greeks()
    #greeks_expo = list(map(ptf.compute_greeks2, ['delta', 'gamma', 'vega', 'theta']))


    vega_bin = bin.vega_binary('long',type='call', S=100, K=110, T=1)

    asian_put = mcm.asian_mcm(101, 100, 12, 1000, 2, 'p')

    put_vanilla = euro.put_european(100, 100, 1)

    put_up_out = mcm.put_up_out(100,100, 125,100,1000,1)
    put_down_out = mcm.put_down_out(100, 100, 125, 100, 1000, 1)
