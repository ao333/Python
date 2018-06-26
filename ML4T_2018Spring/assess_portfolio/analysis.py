import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data, plot_data


def assess_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), syms=['GOOG', 'AAPL', 'GLD', 'XOM'],
                     allocs=[0.1, 0.2, 0.3, 0.4], sv=1000000, rfr=0.0, sf=252.0, gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Daily normalized and weighted portfolio value
    port_val = (prices/prices.iloc[0] * allocs * sv).sum(axis=1)

    # Portfolio statistics
    dr = (port_val / port_val.shift(1)) - 1  # dr.ix[0] = 0
    cr = (port_val[-1] / port_val[0]) - 1
    adr = dr.mean()
    sddr = dr.std()
    sr = np.sqrt(sf) * (adr-rfr) / sddr
    ev = port_val[-1]

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp = df_temp / df_temp.ix[0]
        df_temp.plot()
        plt.title("Portfolio Performance vs SPY")
        plt.xlabel('Daily')
        plt.ylabel('Normalized Price')
        plt.savefig('Portfolio Performance vs SPY')
        plt.show()

    return cr, adr, sddr, sr, ev


def test_code():
    start_date = dt.datetime(2009, 1, 1)
    end_date = dt.datetime(2010, 1, 1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd=start_date, ed=end_date, syms=symbols, allocs=allocations,
                                             sv=start_val, rfr=risk_free_rate, sf=sample_freq, gen_plot=True)

    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    test_code()
