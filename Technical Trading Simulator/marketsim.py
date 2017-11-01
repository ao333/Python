import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data
import matplotlib.pyplot as plt


def author():
    return 'Ao'

def compute_daily_returns(daily_pos_df):
    """Compute and return the daily return values."""
    # compute daily returns for row 1 onwards
    # df_temp = daily_pos_df.sum(axis=1)
    daily_returns = (daily_pos_df / daily_pos_df.shift(1)) - 1
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


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000):
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True,
                            usecols=['Date', 'Symbol', 'Order', 'Shares'])

    start_dt = orders_df.index[0]
    last_dt = orders_df.index[-1]
    unique_symbols = orders_df.drop_duplicates('Symbol')['Symbol'].tolist()

    # Step 1 - Prices Table
    prices_df = get_data(unique_symbols, pd.date_range(start_dt, last_dt), True, 'Adj Close')
    prices_df['Cash'] = 1

    # Step 2 - Trades Table
    trades_df = prices_df.copy()
    for symbol in unique_symbols:
        trades_df[symbol + '_Trade'] = 0
    trades_df['Cash_Trade'] = 0

    for index, row in orders_df.iterrows():
        symbol = row['Symbol']
        buy_or_sell = row['Order']
        units = row['Shares']
        units = units if (buy_or_sell == 'BUY') else units * -1
        price = trades_df.ix[index, symbol]
        cash_units = units * price * -1
        trades_df.ix[index, symbol + '_Trade'] += units
        trades_df.ix[index, 'Cash_Trade'] += cash_units

    trades_df.ix[start_dt, 'Cash_Trade'] += start_val
    # Step 3 - Holdings Table
    holdings_df = trades_df.copy()
    holdings_df['Cash_Holdings'] = holdings_df['Cash_Trade'].cumsum(axis=0)
    for symbol in unique_symbols:
        holdings_df[symbol + '_Holdings'] = holdings_df[symbol + '_Trade'].cumsum(axis=0)

    # Step 4- Values Tables
    values_df = holdings_df.copy()
    values_df['Value'] = 0
    for index, row in values_df.iterrows():

        cash_val = values_df.ix[index, 'Cash_Holdings']
        stocks_val_total = 0
        for symbol in unique_symbols:
            price = values_df.ix[index, symbol]
            holding = values_df.ix[index, symbol + '_Holdings']
            values = price * holding
            stocks_val_total += values
        values_df.ix[index, 'Value'] = stocks_val_total + cash_val
        # values_df.to_csv(orders_file+"_result.csv")
    return values_df.ix[:, 'Value']


def test_code():
    # Define input parameters

    of = "./orders/orders.csv"
    # of = "./orders/orders2.csv"
    # of = "./orders/orders-short.csv"
    of = "./orders/orders_joo.csv"
    # of = "./orders/orders.csv"
    of = "./orders/orders-short.csv"
    # of = "./orders/orders3.csv"
    # of = "./orders/orders2.csv"
    of = "./orders/orders.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get portfolio stats
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    orders_df = pd.read_csv(of, index_col='Date', parse_dates=True,
                            usecols=['Date', 'Symbol', 'Order', 'Shares'])
    unique_symbols = orders_df.drop_duplicates('Symbol')['Symbol'].tolist()
    # return cr, adr, sddr, sr, ev
    daily_rets = compute_daily_returns(portvals)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [compute_cr(portvals), compute_avg_daily_return(daily_rets),
                                                           compute_sddr(daily_rets),
                                                           compute_sr(daily_rets, 0, 252)]

    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [compute_cr(portvals),
                                                                           compute_avg_daily_return(daily_rets),
                                                                           compute_sddr(daily_rets),
                                                                           compute_sr(daily_rets, 0, 252)]

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])


if __name__ == "__main__":
    test_code()
