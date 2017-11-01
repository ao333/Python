import numpy
import pandas as pd
import math as m


# Moving Average
def calc_sma(df, n):
    col_name = 'SMA_' + str(n)
    sma = pd.Series(pd.rolling_mean(df['Adj Close'], n), name=col_name)
    df = df.join(sma)
    df['Adj Close / SMA'] = df[col_name] / df['Adj Close']
    return df


# Exponential Moving Average
def calc_ema(df, n):
    EMA = pd.Series(pd.ewma(df['Adj Close'], span=n, min_periods=n - 1), name='EMA_' + str(n))
    df = df.join(EMA)
    return df


# Bollinger Bands
def calc_bb_bands(df, n):
    MA = pd.Series(pd.rolling_mean(df['Adj Close'], n))
    MSD = pd.Series(pd.rolling_std(df['Adj Close'], n))
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name='BollingerB_' + str(n))
    df = df.join(B1)
    b2 = (df['Adj Close'] - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name='Bollinger%b_' + str(n))
    df = df.join(B2)
    return df


# MACD, MACD Signal and MACD difference
def calc_macd(df, n_fast, n_slow):
    EMAfast = pd.Series(pd.ewma(df['Adj Close'], span=n_fast, min_periods=n_slow - 1))
    EMAslow = pd.Series(pd.ewma(df['Adj Close'], span=n_slow, min_periods=n_slow - 1))
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(pd.ewma(MACD, span=9, min_periods=8), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df
