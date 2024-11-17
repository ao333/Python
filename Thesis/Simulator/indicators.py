import pandas as pd
import matplotlib.pyplot as plt
from util import get_data


def getSMA(v, n):
    return v.rolling(n).mean()


def getSTD(v, n):
    return v.rolling(n).std()


def bollinger_bands(rm, rstd):
    upper_band = rm + rstd * 2
    lower_band = rm - rstd * 2
    return upper_band, lower_band


def momentum_value(v, n):
    mt = (v/v.shift(n)) - 1
    return mt


# Death Cross indicates when moving average of different days cross other moving averages
def getCross(data, window=25):
    # data: Series, contains prices with date indices
    # crossWindow: int, look-back window for simple moving average
    value = 1 / (1 + data / getSMA(data, window))  # ranges from 0 to 1, usually around 0.5
    # set sign = 1 (buy) for prices going up in time and sma going down.
    # set sign = -1 (sell) for the opposite situation.
    sign = (1 * ((data.diff(periods=window) > 0) & (getSMA(data, window).diff(periods = window) < 0)) | \
            -1 * ((data.diff(periods=window) < 0) & (getSMA(data, window).diff(periods = window) > 0)))
    indicator = sign * value
    indicator.name = 'crossIndicator'  # set column name
    # returns Series with values ~ 0.5/-0.5 (buy/sell) on days where the crossover occurs at the valleys/peaks of the
    # slow oscillations. On days where the slopes of the time-series and its sma have the same sign, the value is set to
    # 0. On days where the slopes have opposite sign but a crossover does not occur, values between -1 and 1 are returned
    return indicator


if __name__ == "__main__":

    dates = pd.date_range('2010-07-16', '2018-08-28')
    symbols = '$NQ'
    smaN = 60
    stdN = 20
    mtN = 10
    crossN = 25

    df_init = get_data([symbols], dates, False)
    df_fill = df_init.ffill().bfill()
    df = df_fill / df_fill.iloc[0, ]

    sma = getSMA(df[symbols], smaN)
    std = getSTD(df[symbols], stdN)
    upper_band, lower_band = bollinger_bands(sma, std)
    mt = momentum_value(df[symbols], mtN)
    cross = getCross(df[symbols], crossN)

    div = df.divide(sma, axis='index')
    re = df.join(sma, lsuffix='_Normalized Price', rsuffix='_SMA').join(div, lsuffix='', rsuffix='_NormPrice/SMA')
    re.columns = ['Normalized Price', 'SMA', 'Normalized Price/SMA']
    ax = re.plot(title="Normalized Price & SMA", fontsize=12, lw=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")

    re = df.join(sma, lsuffix='_Normalized Price', rsuffix='_SMA').join(upper_band, lsuffix='_',
                rsuffix='_upperband').join(lower_band, lsuffix='_', rsuffix='_lowerband')
    re.columns = ['Normalized Price', 'SMA', 'Upper Bands', 'Lower Bands']
    ax = re.plot(title="Normalized Price & Bollinger bands", fontsize=12, lw=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")

    re = df.join(mt, lsuffix='_Normalized Price', rsuffix='_Momentum')
    re.columns = ['Normalized Price', 'Momentum']
    ax = re.plot(title="Normalized Price & Momentum", fontsize=12, lw=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized Price")

    plt.figure()
    (1 / (1 + df[symbols] / getSMA(df[symbols], crossN))).plot(title='Crossover Value')
    plt.xlabel('Date')
    plt.ylabel('Crossover Value')

    plt.figure()
    cross.plot(title='Crossover Indicator')
    plt.xlabel('Date')
    plt.ylabel('Crossover Ratio')

    plt.figure()
    std.plot(title="Standard Deviation", label='STD')
    plt.xlabel('Date')
    plt.ylabel('Price')

    plt.show()
