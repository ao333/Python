import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as spo

from util import get_data


def compute_daily_returns(daily_pos_df):
    """Compute and return the daily return values."""
    daily_returns = daily_pos_df.copy()
    # compute daily returns for row 1 onwards
    df_temp = daily_pos_df.sum(axis=1)
    daily_returns = (df_temp / df_temp.shift(1)) - 1
    # daily_returns.ix[0, :] = 0  # set daily returns for row 0 to 0
    return daily_returns


def compute_normalized(df):
    normalized = df[0:] / df.ix[0]
    return normalized


def compute_alloced(normalized, syms, allocs):
    alloced = normalized[syms] * allocs
    return alloced


def compute_pos_vals(alloced, sv):
    return alloced * sv


def compute_avg_daily_return(daily_returns):
    return daily_returns.mean()


def compute_sddr(daily_returns):
    return daily_returns.std()


def compute_cr(port_val):
    return (port_val[-1] / port_val[0]) - 1


def compute_sr(daily_returns, rf, sf):
    return ((daily_returns - rf).mean() / daily_returns.std(ddof=1)) * np.sqrt(sf)


def assess_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), \
                     syms=['GOOG', 'AAPL', 'GLD', 'XOM'], \
                     allocs=[0.1, 0.2, 0.3, 0.4], \
                     sv=1000000,
                     rfr=0.0,
                     sf=252.0, \
                     gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    normalized = compute_normalized(prices_all)
    # print normalized
    alloced = compute_alloced(normalized, syms, allocs)

    pos_vals = compute_pos_vals(alloced, sv)
    daily_rets = compute_daily_returns(pos_vals)
    port_val = pos_vals.sum(axis=1)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = [compute_cr(port_val), compute_avg_daily_return(daily_rets), compute_sddr(daily_rets),
                         compute_sr(daily_rets, rfr, sf)]  # add code here to compute stats

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_normalized = compute_normalized(df_temp)
        df_normalized.plot(title='Daily Portfolio Value and SPY')
        plt.show()
        pass

    # compute end value
    ev = sv

    return cr, adr, sddr, sr, ev


def f(X):
    Y = (X - 1.5) ** 2 + 0.5
    print "X = {}, Y= {}".format(X, Y)
    return Y


def error(allocs, syms, prices):
    """Compute error between given line model and observed data.
    Parameters
    -----------
    line: tuple/list/array (C0, C1) where C0 is slope and C1 is the Y-intercept
    data: 2D array where each row is a point (x,y)

    Returns error as a single real value

    """
    normalized = compute_normalized(prices)
    alloced = compute_alloced(normalized, syms, allocs)
    pos_vals = compute_pos_vals(alloced, 1000)
    dailiy_rets = compute_daily_returns(pos_vals)
    sr = compute_sr(dailiy_rets, 0, 252)
    return sr * -1


def fit_line(data, error_func):
    # Generate Initial Guess
    l = np.float32([0, np.mean(data[:, 1])])  # slope = 0 intercept = mean(y values)
    # Plot initial guess(optional)
    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial Guess")
    result = spo.minimize(error_func, l, args=(data,), method='SLSQP', options={'disp': True})
    return result.x


def optimize_sr(syms, prices):
    # generate initial guess
    count = len(syms)
    avg = 1 / float(count)
    alloc = np.full((1, count), avg)
    bounds = [(0.0, 1.0)] * count
    const = ({'type': 'eq', 'fun': lambda inputs: 1 - np.sum(inputs)})
    result = spo.minimize(error, alloc, args=(syms, prices,), method='SLSQP', options={'disp': True}, bounds=bounds,
                          constraints=const)
    return result.x

def optimize_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1), \
                       syms=['GOOG', 'AAPL', 'GLD', 'XOM'], gen_plot=False):
    sv = 1000000
    rfr = 0.0
    sf = 252.0

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    allocs = optimize_sr(syms, prices)

    # Get daily portfolio value
    normalized = compute_normalized(prices_all)
    # print normalized
    alloced = compute_alloced(normalized, syms, allocs)

    pos_vals = compute_pos_vals(alloced, sv)
    daily_rets = compute_daily_returns(pos_vals)
    port_val = pos_vals.sum(axis=1)

    cr, adr, sddr, sr = [compute_cr(port_val), compute_avg_daily_return(daily_rets), compute_sddr(daily_rets),
                         compute_sr(daily_rets, rfr, sf)]  # add code here to compute stats

    if gen_plot:
        df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_normalized = compute_normalized(df_temp)
        df_normalized.plot(title='Daily Portfolio Value and SPY')
        plt.show()
        pass

    return allocs, cr, adr, sddr, sr


def test_code():
    # This function WILL NOT be called by the auto grader
    # Do not assume that any variables defined here are available to your function/code
    # It is only here to help you set up and test your code

    # Define input parameters

    start_date = dt.datetime(2009, 1, 1)
    end_date = dt.datetime(2010, 1, 1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, \
                                                        syms=symbols, \
                                                        gen_plot=True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr

    """
    Start Date: 2010-01-01
    End Date: 2010-12-31
    Symbols: ['GOOG', 'AAPL', 'GLD', 'XOM']
    Optimal allocations: [  5.38105153e-16   3.96661695e-01   6.03338305e-01  -5.42000166e-17]
    Sharpe Ratio: 2.00401501102
    Volatility (stdev of daily returns): 0.0101163831312
    Average Daily Return: 0.00127710312803
    Cumulative Return: 0.360090826885
    """
    start_date = dt.datetime(2010, 1, 1)
    end_date = dt.datetime(2010, 12, 31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date, \
                                                        syms=symbols, \
                                                        gen_plot=True)

    # Print statistics
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
