import os
import pandas as pd


# Return CSV file path given ticker symbol
def symbol_to_path(symbol, base_dir=None):
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", '../data/')
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))


# Takes an order, a filename, and a base directory and writes a csv order file
def createOrder(order, fname, base_dir='', symbol='AAPL', unit=200):
    # order: Series, consists of the values -2, -1, 0, 1, 2 denoting buying/selling of 200 or 400 shares or just holding
    # fname: string, the filename
    # base_dir: string, the base directory
    # symbol: string, the stock to trade
    # unit: integer, number of stocks to trade per unit in the order Series

    f = open(symbol_to_path(fname, base_dir), 'w')
    f.write('Date,Symbol,Order,Shares\n')
    for ind, val in order.iteritems():  # iterator over the timestamp indices
        # and values in 'order'
        if val == 1:
            f.write('{},{},BUY,{}\n'.format(ind.date(), symbol, unit))
        elif val == -1:
            f.write('{},{},SELL,{}\n'.format(ind.date(), symbol, unit))
        elif val == 2:
            f.write('{},{},BUY,{}\n'.format(ind.date(), symbol, 2 * unit))
        elif val == -2:
            f.write('{},{},SELL,{}\n'.format(ind.date(), symbol, 2 * unit))


# Read stock mc3_p1.data (adjusted close) for given symbols from CSV files
def get_data(symbols, dates, addSPY=True, colname='Adj Close'):
    df = pd.DataFrame(index=dates)
    if addSPY and '$NQ' not in symbols:  # add SPY for reference, if absent
        symbols = ['$NQ'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',
                              parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == '$NQ':  # drop dates SPY did not trade
            df = df.dropna(subset=["$NQ"])

    return df


def get_orders_data_file(basefilename):
    return open(os.path.join(os.environ.get("ORDERS_DATA_DIR", 'orders/'), basefilename))


def get_learner_data_file(basefilename):
    return open(os.path.join(os.environ.get("LEARNER_DATA_DIR", 'data/'), basefilename), 'r')


def get_robot_world_file(basefilename):
    return open(os.path.join(os.environ.get("ROBOT_WORLDS_DIR", 'testworlds/'), basefilename))
