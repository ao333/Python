import numpy as np
import pandas as pd
from util import get_data


# def compute_portvals(orders_file = "./orders/orders.csv", start_val=1000000, commission=9.95, impact=0.005):
def compute_portvals(orders, start_val=1000000, commission=9.95, impact=0.005):
    # df = pd.read_csv(orders_file, index_col = 'Date', parse_dates = True, na_values = ['nan'])
    df = file_df(orders)
    start_date = min(df.index)
    end_date = max(df.index)

    symbols = []
    for i, row in df.iterrows():
        if row['Symbol'] not in symbols:
            symbols.append(row['Symbol'])

    prices_symbol = get_data(symbols, pd.date_range(start_date, end_date))
    for symbol in symbols:
        prices_symbol[symbol + ' Shares'] = pd.Series(0, index=prices_symbol.index)
        prices_symbol['Port Val'] = pd.Series(start_val, index=prices_symbol.index)
        prices_symbol['Cash'] = pd.Series(start_val, index=prices_symbol.index)

    for i, row in df.iterrows():
        symbol = row['Symbol']
        if row['Order'] == 'BUY':
            prices_symbol.loc[i:, symbol + ' Shares'] = prices_symbol.loc[i:, symbol + ' Shares'] + row['Shares']
            prices_symbol.loc[i:, 'Cash'] -= prices_symbol.loc[i, symbol] * row['Shares'] * (1+impact) + commission
        if row['Order'] == 'SELL':
            prices_symbol.loc[i:, symbol + ' Shares'] = prices_symbol.loc[i:, symbol + ' Shares'] - row['Shares']
            prices_symbol.loc[i:, 'Cash'] += prices_symbol.loc[i, symbol] * row['Shares'] * (1-impact) - commission

    for i, row in prices_symbol.iterrows():
        shares_val = 0
        for symbol in symbols:
            shares_val += prices_symbol.loc[i, symbol + ' Shares'] * row[symbol]
            prices_symbol.loc[i, 'Port Val'] = prices_symbol.loc[i, 'Cash'] + shares_val

    return prices_symbol.loc[:, 'Port Val']


def compute_portfolio(allocs, prices, sv=1):
    normed = prices / prices.iloc[0, ]
    alloced = normed * allocs
    pos_vals = alloced * sv
    port_val = pos_vals.sum(axis=1)
    return port_val


def compute_portfolio_stats(port_val):
    daily_ret = (port_val/port_val.shift(1)) - 1
    cr = (port_val[-1]/port_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    sr = np.sqrt(252.0) * ((daily_ret - 0.0).mean() / sddr)
    return cr, adr, sddr, sr


def file_df(file):
    symbol = []
    order = []
    share = []
    for i in range(len(file.index)):
        symbol.append('JPM')
        if file['orders'][i] > 0:
            order.append('BUY')
            share.append(file['orders'][i])
        elif file['orders'][i] < 0:
            order.append('SELL')
            share.append(-file['orders'][i])
    df_symbol = pd.DataFrame(data=symbol, index=file.index, columns=['Symbol'])
    df_order = pd.DataFrame(data=order, index=file.index, columns=['Order'])
    df_share = pd.DataFrame(data=share, index=file.index, columns=['Shares'])
    df_result = df_symbol.join(df_order).join(df_share)
    return df_result


def author():
    return 'Ao Shen'


if __name__ == "__main__":
    of = "./orders./orders-01.csv"
    sv = 1000000

    # Get portfolio stats
    portvals = compute_portvals(orders_file=of, start_val=sv)
    df = pd.read_csv(of, index_col='Date', parse_dates=True, na_values=['nan'])
    start_date = min(df.index)
    end_date = max(df.index)

    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]
    portvals_SPX = compute_portfolio([1.0], prices_SPX)
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = compute_portfolio_stats(portvals_SPX)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(portvals)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX : {}".format(sharpe_ratio_SPX)
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX : {}".format(cum_ret_SPX)
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX : {}".format(std_daily_ret_SPX)
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX : {}".format(avg_daily_ret_SPX)
    print "Final Portfolio Value: {}".format(portvals[-1])
