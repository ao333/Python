import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data
from Simulator.marketsim import compute_portvals, compute_portfolio_stats
from indicators import getSMA, getSTD, bollinger_bands, momentum_value


class ManualStrategy(object):

    def __init__(self):
        self.long_entry = []
        self.short_entry = []

    def testPolicy(self, symbol="$NQ", sd=dt.datetime(2010, 7, 16), ed=dt.datetime(2018, 8, 28), sv = 100000):
        dates = pd.date_range(sd, ed)
        symbols = symbol
        n = 20
        df = get_data([symbols], dates)
        df = df.ffill().bfill()

        rm = getSMA(df[symbols], n)
        rstd = getSTD(df[symbols], n)
        upper_band, lower_band = bollinger_bands(rm, rstd)
        mt = momentum_value(df[symbols], n)

        pos = 0
        share = []
        date = []
        # -1,0,1: short, out, long
        for i in range(len(upper_band.index)):
            if df[symbols][i-1] > upper_band.loc[upper_band.index[i-1]] \
                    and df[symbols][i] < upper_band.loc[upper_band.index[i]] and pos != -1000:
                share.append(-1000-pos)
                date.append(df.index[i])
                pos = -1000
                self.short_entry.append(df.index[i])

            elif df[symbols][i-1] > rm.loc[upper_band.index[i-1]]\
                    and df[symbols][i] < rm.loc[upper_band.index[i]] and pos == -1000:
                share.append(1000)
                date.append(df.index[i])
                pos = 0

            elif df[symbols][i-1] < lower_band.loc[upper_band.index[i-1]]\
                    and df[symbols][i] > lower_band.loc[upper_band.index[i]] and pos != 1000:
                share.append(1000-pos)
                date.append(df.index[i])
                pos = 1000
                self.long_entry.append(df.index[i])

            elif df[symbols][i-1] < rm.loc[upper_band.index[i-1]]\
                    and df[symbols][i] > rm.loc[upper_band.index[i]] and pos == 1000:
                share.append(-1000)
                date.append(df.index[i])
                pos = 0

            elif mt.loc[upper_band.index[i]] > 0.6 and pos != 1000:
                share.append(1000-pos)
                date.append(df.index[i])
                pos = 1000
                self.long_entry.append(df.index[i])

            elif mt.loc[upper_band.index[i]] < -0.5 and pos != -1000:
                share.append(-1000-pos)
                date.append(df.index[i])
                pos = -1000
                self.short_entry.append(df.index[i])

        if pos != 0:
            share.append(-pos)
            date.append(df.index[len(df.index)-1])

        df_trades = pd.DataFrame(data=share, index=date, columns=['orders'])
        return df_trades

    def benchMark(self, symbol="$NQ", sd=dt.datetime(2010, 7, 16), ed=dt.datetime(2018, 8, 28), sv = 100000):
        dates = pd.date_range(sd, ed)
        symbols = symbol
        df = get_data([symbols], dates)
        share = [1000, -1000]
        date = [df.index[0], df.index[len(df.index)-1]]
        df_bchm = pd.DataFrame(data=share, index=date, columns=['orders'])
        return df_bchm


if __name__ == "__main__":
    start_date = '2010-07-16'
    end_date = '2018-08-28'
    symbols = '$BTC'

    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbols, start_date, end_date, 100000)
    port_vals = compute_portvals(orders=df_trades, start_val=100000, commission=9.95, impact=0.005)
    df_bm = ms.benchMark(symbols, start_date, end_date, 100000)
    port_vals_bm = compute_portvals(orders=df_bm, start_val=100000, commission=9.95, impact=0.005)

    cr, adr, sdr, sr = compute_portfolio_stats(port_vals)
    cr_bm, adr_bm, sdr_bm, sr_bm = compute_portfolio_stats(port_vals_bm)

    # Compare portfolio against SPX
    print "Date Range(In Sample): {} to {}".format(start_date, end_date)
    print
    print "In Sample Cumulative Return of Portfolio: {}".format(cr)
    print "In Sample Cumulative Return of Benchmark: {}".format(cr_bm)
    print
    print "In Sample Standard Deviation of Portfolio: {}".format(sdr)
    print "In Sample Deviation of Benchmark: {}".format(sdr_bm)
    print
    print "In Sample Average Daily Return of Portfolio: {}".format(adr)
    print "In Sample Average Daily Return of Benchmark: {}".format(adr_bm)
    print
    print "In Sample Sharpe Ratio of Portfolio: {}".format(sr)
    print "In Sample Sharpe Ratio of Benchmark: {}".format(sr_bm)
    print
    print "In Sample Final Portfolio Value: {}".format(port_vals[-1])
    print "In Sample Final Benchmark Value: {}".format(port_vals_bm[-1])

    port_vals_bm_norm = port_vals_bm / port_vals_bm.loc[0, ]
    port_vals_norm = port_vals / port_vals.loc[0, ]
    port_vals_bm_norm = port_vals_bm_norm.to_frame()
    port_vals_norm = port_vals_norm.to_frame()
    long_entry = ms.long_entry
    short_entry = ms.short_entry

    f1 = plt.figure(1)
    re = port_vals_bm_norm.join(port_vals_norm, lsuffix='_Nasdaq 100', rsuffix='_BitCoin')
    re.columns = ['Benchmark', 'Value of the best possible portfolio']
    ax = re.plot(title="Nasdaq 100 and Bitcoin Buy (Green) and Sell (Red) Signals", fontsize=12, color=["blue", "black"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Signal Strength")
    ymin, ymax = ax.get_ylim()
    plt.vlines(long_entry, ymin, ymax, color='g')
    plt.vlines(short_entry, ymin, ymax, color='r')
    f1.show()


    f2 = plt.figure(2)
    re = port_vals_bm_norm.join(port_vals_norm, lsuffix='_Nasdaq 100', rsuffix='_BitCoin')
    re.columns = ['Benchmark', 'Value of The Best Possible Portfolio']
    ax = re.plot(title="Nasdaq 100 and Bitcoin Buy (Green) and Sell (Red) Signals", fontsize=12, color=["blue", "black"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio")
    f2.show()

    # Plot code for out of sample data
    start_date = '2010-07-16'
    end_date = '2018-08-28'
    symbols = '$BTC'

    ms = ManualStrategy()
    df_trades = ms.testPolicy(symbols, start_date, end_date, 100000)
    port_vals = compute_portvals(orders=df_trades, start_val=100000, commission=9.95, impact=0.005)
    df_bm = ms.benchMark(symbols, start_date, end_date, 100000)
    port_vals_bm = compute_portvals(orders=df_bm, start_val=100000, commission=9.95, impact=0.005)

    cr, adr, sdr, sr = compute_portfolio_stats(port_vals)
    cr_bm, adr_bm, sdr_bm, sr_bm = compute_portfolio_stats(port_vals_bm)

    # Compare portfolio against SPX
    print "Date Range(Out of Sample): {} to {}".format(start_date, end_date)
    print
    print "Out of Sample Cumulative Return of Portfolio: {}".format(cr)
    print "Out of Sample Cumulative Return of Benchmark: {}".format(cr_bm)
    print
    print "Out of Sample Standard Deviation of Portfolio: {}".format(sdr)
    print "Out of Sample Deviation of Benchmark: {}".format(sdr_bm)
    print
    print "Out of Sample Average Daily Return of Portfolio: {}".format(adr)
    print "Out of Sample Average Daily Return of Benchmark: {}".format(adr_bm)
    print
    print "Out of Sample Sharpe Ratio of Portfolio: {}".format(sr)
    print "Out of Sample Sharpe Ratio of Benchmark: {}".format(sr_bm)
    print
    print "Out of Sample Final Portfolio Value: {}".format(port_vals[-1])
    print "Out of Sample Final Benchmark Value: {}".format(port_vals_bm[-1])

    port_vals_bm_norm = port_vals_bm / port_vals_bm.loc[0, ]
    port_vals_norm = port_vals / port_vals.ix[0, ]
    port_vals_bm_norm = port_vals_bm_norm.to_frame()
    port_vals_norm = port_vals_norm.to_frame()

    f3 = plt.figure(3)
    re = port_vals_bm_norm.join(port_vals_norm, lsuffix='_benchmark', rsuffix='_portfolio')
    re.columns = ['Benchmark', 'Value of the best possible portfolio']
    ax = re.plot(title="Benchmark vs Best Possible Portfolio (Out Sample, Normed)", fontsize=12, color=["blue", "black"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio")
    f3.show()

    plt.show()

