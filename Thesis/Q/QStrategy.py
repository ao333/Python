import time
import util as ut
import numpy as np
import pandas as pd
import datetime as dt
import Q as ql
import matplotlib.pyplot as plt
from RF.indicators import createOrder
from RF.marketsim import testcode_marketsim
from RF.rule_based import getSmaIndicator, getMomIndicator, getCrossIndicator, plotVline


class StrategyLearner(object):

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.unit = 200  # number of stocks to trade per unit in the order Series
        self.holdTime = 21  # number of days to wait after buying / selling
        self.scaleForPlot = 0.1  # scale factor for reward to enable co-plotting

    def dailyTradingStrategy(self, signal, currOrder, date, endDate, day, holdTime):
        """
        @Summary: Creates an order, on a day by day basis, from a trading signal using 1 of 2 possible strategies
        @parameter signal: integer, sell = -1, hold = 0, buy = 1
        @parameter currOrder: integer, short = 0, noPos = 1, long = 2
        @parameter date: datetime object, today's date
        @parameter endDate: datetime object, last date of trading period
        @parameter day: integer, indexes current date in trading period
        @returns order: integer, values -2, -1, 0, 1, 2 denote selling/buying of 1 or 2 trading units of shares
        (depending on the strategy selected by commenting out below), or just holding
        @returns currOrder: integer, updated current position
        @returns day: integer, indexes next date
        """
        order = 0  # no trade if signal = 0 and date != endDate (i.e. if statement below falls through without execution)
        if date == endDate:  # if we reach the end of the trading period, redeem all outstanding positions
            order = -currOrder
            day += 1  # allows to exit the inner 'while' loop in addEvidence method

        # BEGIN OF +/- 1 UNIT TRADING STRATEGY 1
        elif (currOrder < 1) and (signal == 1):  # current order status is not long and signal to buy is given
            order = 1  # buy 200
            currOrder += order
            day += holdTime  # after buying wait for hold period
        elif (currOrder > -1) and (signal == -1):  # current order status is not short and signal to sell is given
            order = -1  # sell 200
            currOrder += order
            day += holdTime  # after selling wait for hold period
        else:
            day += 1  # if signal = 0
            
        return order, currOrder, day

    def getIndicators(self, prices):
        """
        @Summary: Uses 3 indicators (can be changed) to construct a DataFrame
        @parameter prices: Series, contains price with date indices
        @returns df: DataFrame, whose columns contain indicators without NaNs
        """
        self.numSteps = np.asarray([10, 10, 10])
        # each element corresponds to number of steps or bins used to discretize each indicator.
        # Each element can be independently varied
        smaIndicator, smaWindow = getSmaIndicator(prices)
        momIndicator, momWindow = getMomIndicator(prices)
        crossIndicator, crossWindow = getCrossIndicator(prices)
        df = pd.DataFrame(index=prices.index)
        df = df.join([smaIndicator, momIndicator, crossIndicator])
        df.dropna(inplace=True)  # drop any rows having some NaN values
        return df
        
    def getState(self, currOrder_base0, dailyDiscrete, numSteps):
        """
        @Summary: combines current order and discretized indicators on a particular day to get a unique state
        @parameter currOrder_base0: integer, currOrder incremented by 1 so that short = 0, noPos = 1, long = 2
        @parameter dailyDiscrete: ndarray, arbitrary number of discretized indicators for one day
        @parameter numSteps: ndarray, corresponds to number of steps or bins used to discretize each indicator.
        @returns state: integer
        """
        state = currOrder_base0 * np.prod(numSteps)  # contribution to state from the 'most significant digit
        for i in range(len(dailyDiscrete)):  # iterate over each discretized indicator
            if i == len(dailyDiscrete) - 1:  # 'least significant digit'
                state += dailyDiscrete[i]
            else:  # contribution to state from digits other than the 'most significant' and 'least significant'
                state += dailyDiscrete[i] * np.prod(numSteps[i + 1:])
        return int(state)

    def addEvidence(self, symbol="$NQ", sd=dt.datetime(2003, 1, 1), ed=dt.datetime(2018, 1, 1), sv=10000):

        def discretize(df, numSteps = np.asarray([10, 10, 10])):
            """
            @Summary: Converts continuous indicators of a stock into discretized indicators by deciding how many elements
            will be in each bin (say n), sorting each indicator, & setting the thresholds for every nth element
            @parameter df: DataFrame, continuous indicators
            @parameter numsteps: ndarray, each element corresponds to number of steps or bins used
            to discretize each indicator. Each element can be independently varied.
            @returns dfThresholded: DataFrame, discretized indicators
            @returns thresholdList: List, of sublists of increasing values of thresholds used for binning.
            Each sublist corresponds to one indicator
            """
            dfSorted = df.copy()  # make copy so that we can sort in-place without changing the original dataFrame
            counter = 0  # iterator for numSteps desired for each indicator
            thresholdList = []
            for col in list(dfSorted):  # for each column of indicators
                # stepSize = number of elements in each bin
                stepSize = int(round(len(df) / float(numSteps[counter])))
                threshold = np.zeros(numSteps[counter])  # appropriate number of thresholds initialized
                dfSorted.sort_values(col, inplace=True)  # sort on selected col
                for i in range(numSteps[counter]):
                    if i == (numSteps[counter] - 1):  # for last bin, thresh=max
                        threshold[i] = dfSorted[col].values[-1]  # or last value
                        # set all remaining elements in the sorted column of indicators to be in the last bin or step
                        dfSorted[col].ix[i * stepSize :] = i
                    else:  # set threshold = every stepSize-th element
                        threshold[i] = dfSorted[col].values[(i + 1) * stepSize]
                        # set all prior elements in the sorted column of indicators to the appropriate bin or step number
                        dfSorted[col].ix[i * stepSize : (i + 1) * stepSize] = i

                thresholdList.append(threshold)  # append ndarray of thresholds
                counter += 1

            dfThresholded = dfSorted.sort_index()  # restore sort on df indices
            return dfThresholded, thresholdList

        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols, prices is a DataFrame
        # prices[symbol] converts prices DataFrame to Series
        df = self.getIndicators(prices[symbol])  # df of continuous indicators
        dfThresholded, thresholdList = discretize(df, self.numSteps)
        self.thresholdList = thresholdList  # save for reference by testPolicy
        # initialize QLearner. 3 accounts for no-position, short and long states and for hold, sell and buy actions
        self.learner = ql.QLearner(num_states=3 * np.prod(self.numSteps), num_actions=3, dyna=200, verbose=False)
        
        maxIteration = 10  # iterations to try for convergence to a policy
        cumReturn = np.zeros(maxIteration)
        iteration = 0
        numDays = len(df)  # maximum number of days in the trading period
        start = time.time()
        if self.verbose:
            plt.figure(figsize=(11, 4))
                    
        while iteration < maxIteration:
            order = pd.DataFrame(np.zeros(len(df)), index=df.index, columns=['number of units'])
            currOrder = 0  # current order status: short= -1, noPos = 0, long = 1
            day = 0  # start of period for which indicators are available
            rList = []  # list of rewards, used only for plotting
            while day < numDays:
                date = df.index[day]
                dailyDiscrete = dfThresholded.values[day, :]  # today's discretized indicator values
                # get unique state with currOrder incremented by unity so it takes values >= 0
                state = self.getState(currOrder + 1, dailyDiscrete, self.numSteps)
                if day == 0:  # the first day with technical indicators. No reward since we enter with no-position
                    action = self.learner.querysetstate(state)
                    signal = action - 1  # to represent short = -1, hold = 0, long = 1
                    oldPrice = prices.ix[date, 0]  # [date,0] returns a pure number from the prices DataFrame
                else:
                    priceDifference = prices.ix[date, 0] - oldPrice
                    r = currOrder * priceDifference  # reward/penalty for prior selection of the position to be in
                    action = self.learner.query(state, r)
                    signal = action - 1
                    rList.append([date, self.scaleForPlot * r])

                # update current order, current position and index of next date
                order.ix[date, 0], currOrder, day = \
                self.dailyTradingStrategy(signal, currOrder, date, df.index[-1], day, self.holdTime)
                if order.ix[date, 0] != 0:  # anchor oldPrice to when a transaction is made
                    oldPrice = prices.ix[date, 0]
                if day >= numDays:  # if index of next date to trade is outside trading period,
                    order.ix[df.index[-1], 0] = -currOrder  # redeem outstanding positions

            createOrder(order['number of units'], 'QL_based', symbol = symbol, unit = self.unit)
            cumReturn[iteration], portVals = testcode_marketsim('QL_based', sv=sv, leverLimit=False, verbose=False)
            if self.verbose:
                print 'iteration = {}, Cumulative return = {}'.format(iteration, cumReturn[iteration])
            if (iteration > 0) & (cumReturn[iteration] == cumReturn[iteration - 1]):
                iteration += 1
                break
            iteration += 1
        
        stop = time.time()
        if self.verbose:
            print 'Elapsed time for training = {} s'.format( round(stop - start, 2))
            plt.plot([item[0] for item in rList], [item[1] for item in rList], '.g', label='scaled reward')
            plt.plot(prices.ix[df.index] / prices.ix[df.index[0]] - 1, 'k', label='benchmark')
            plt.plot(portVals / portVals[0] - 1, 'g', label='QL-Strat')
            plt.ylabel('normalized return')
            plt.title('Training mc3_p1.data, final cumulative return = {}'.format(cumReturn[iteration - 1]))
            plt.legend(loc='best')
            plotVline(order['number of units'])
            plt.axhline(y=0, color='k', linestyle=':')  # plot horizontal line at y = 0

    def testPolicy(self, symbol="$NQ", sd=dt.datetime(2018, 1, 2), ed=dt.datetime(2018, 9, 3), sv=10000):
        """
        @Summary: uses the existing Q-table policy and tests it against new data
        @returns cumReturn: float, cumulative return of portfolio obtained by trading with the learnt Q-table policy
        """
        syms=[symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols without SPY
        # prices[symbol] converts prices DataFrame to Series
        df = self.getIndicators(prices[symbol])
            
        numIndicators = len(self.thresholdList)
        order = pd.DataFrame(np.zeros(len(df)), index=df.index, columns=['number of units'])
        currOrder = 0  # current order status: short= -1, noPos = 0, long = 1
        rList = []  # list of rewards, used only for plotting
        numDays = len(df)
        day = 0
        oldPrice = prices.ix[df.index[0]]  # to calculate reward, get price on first trading day
        # initialize discretized indicators computed below for each day in the trading period
        dailyDiscrete = -1 * np.ones(numIndicators)
        while day < numDays:
            date = df.index[day]
            for i in range(numIndicators):
                step = 0  # initialize to first bin
                # threshold in thresholdList are ordered by ascending values
                for threshold in self.thresholdList[i]:
                    if df.ix[date, i] <= threshold:
                        dailyDiscrete[i] = step
                        break

                    step += 1
                else:  # for loop fell through without identifying a threshold
                    dailyDiscrete[i] = len(self.thresholdList[i]) - 1
            # get unique state with currOrder incremented by unity so it takes values >= 0
            state = self.getState(currOrder + 1, dailyDiscrete, self.numSteps)
            
            priceDifference = prices.ix[date, 0] - oldPrice
            r = currOrder * priceDifference
            rList.append([date, self.scaleForPlot * r])
            # use the QLearner's query that does not update the Q-table
            action = self.learner.querysetstate(state)
            signal = action - 1
            order.ix[date, 0], currOrder, day = \
                self.dailyTradingStrategy(signal, currOrder, date, df.index[-1], day, self.holdTime)
            if order.ix[date, 0] != 0:  # anchor oldPrice to when a
                    oldPrice = prices.ix[date, 0]  # transaction is made
            if day >= numDays:
                    order.ix[df.index[-1], 0] = -currOrder

        createOrder(order['number of units'], 'QL_based', symbol=symbol, unit=self.unit)
        cumReturn, portVals = testcode_marketsim('QL_based', sv=sv, leverLimit=False, verbose=False)

        plt.figure(figsize=(11, 4))
        plt.plot([item[0] for item in rList], [item[1] for item in rList], '.g', label='scaled reward')
        plt.plot(prices / prices.ix[df.index[0]] - 1, 'k', label='benchmark')
        plt.xlim((df.index[0], df.index[-1]))
        plt.plot(portVals / portVals[0] - 1, 'g', label='QL-based')
        plt.legend(loc='best')
        plt.ylabel('Sharpe')
        plt.title('Test data, cumulative return = {}'.format(cumReturn))
        plotVline(order['number of units'])
        plt.axhline(y=0, color='k', linestyle=':')
        plt.show()
        
        return cumReturn


if __name__=="__main__":
    learner = StrategyLearner(True)
    sym = "$NQ"
    repsOuter = 1  # number of times to re-create and re-train the QLearner
    sumCumReturn = 0  # initialize sum of all cumulative returns on the test

    start = time.time()
    for reps in range(repsOuter):
        print 'Outer rep = {}'.format(reps)
        stdate =dt.datetime(2003, 1, 1)
        enddate =dt.datetime(2018, 1, 1)
        # train the learner
        learner.addEvidence(symbol=sym, sd=stdate, ed=enddate, sv=100000)
        # set parameters for testing
        stdate =dt.datetime(2018, 1, 1)
        enddate =dt.datetime(2018, 9, 3)
        # number of times to calculate cumulative return on the test data, after training the learner, in the step above
        repsInner = 1
        for i in range(repsInner):
            cumReturn = learner.testPolicy(symbol=sym, sd=stdate, ed=enddate, sv=100000)
            sumCumReturn += cumReturn

    print 'Average Test cumulative return = {}'.format(sumCumReturn / (repsOuter * repsInner))
    stop = time.time()
    print 'Elapsed total time = {} s'.format(round(stop - start, 4))
