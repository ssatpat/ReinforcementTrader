import datetime as dt
import pandas as pd
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_intraday
from iexfinance.stocks import get_historical_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import QLearner as ql
import features

LONG = 100.0
SHORT = -100.0
YBUY = 0.025
YSELL = -0.025
GAINS_WINDOW=20

class ReinforcementLearner(object):
    def __init__(self, verbose = False, impact=0.0, commission=0.0, numLong = 1000.0, numShort=1000.0):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.bins = None
        self.ql = ql.QLearner(num_states=10000, num_actions=3, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False)
        self.features = None
        self.numLong = numLong
        self.numShort = numShort


    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1)):

        f = features.generateFeatures(symbol, sd, ed)
        self.features = f
        real_features = f[['bb value', 'macd', 'trix', 'volume']]
        discrete_features, self.bins = self.discretize(real_features)
        state_df = self.getState(discrete_features)
        state_df.fillna(value=0, inplace=True)

        daily_returns = f[symbol].pct_change(1)
        self.ql.querysetstate(int(state_df.iloc[0]))

        trades_df = pd.DataFrame(index=f.index)
        trades_df[symbol] = 0.0

        i=0
        converged = False
        position=0.0
        c_ct = 0
        while not converged:
            reward = 0
            total_holdings = 0
            prev_trades = trades_df.copy()

            for index, row in discrete_features.iterrows():
                reward = position*daily_returns.loc[index]*(1-self.impact)
                action = self.ql.query(int(state_df.loc[index]['state']), reward)
                if action == 1 and position == 0.0:
                    #go long
                    position = self.numLong
                    trades_df[symbol][index] = self.numLong
                elif action == 2 and position == 0.0:
                    position = -1.0*self.numShort
                    trades_df[symbol][index] = -1.0*self.numShort
                elif action == 1 and position < 0.0:
                    position = self.numLong
                    trades_df[symbol][index] = 2.0*self.numLong
                elif action == 2 and position > 0.0:
                    position = -1.0*self.numShort
                    trades_df[symbol][index] = -2.0*self.numShort
                elif action == 0:
                    trades_df[symbol][index] = 0.0

            i+=1
            if trades_df.equals(prev_trades):
                c_ct +=1
                if c_ct >5:
                    converged=True
                    print(i)


        return

    def getBenchMarkTrades(self, prices):
        benchmark_orders = pd.DataFrame(columns=['Shares'], index=prices.index)
        benchmark_orders['Shares'].iloc[0] = self.numLong

        benchmark_orders.fillna(0, inplace=True)
        return benchmark_orders['Shares']

    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1)):

        f = features.generateFeatures(symbol, sd, ed)
        #f = self.features
        prices_norm = f[symbol]
        benchmark = self.getBenchMarkTrades(f[symbol])
        real_features = f[['bb value', 'macd', 'trix', 'volume']]

        discrete_indicators = self.discretize_test(real_features)
        state_df = self.getState(discrete_indicators)
        state_df.fillna(value=0, inplace=True)
        self.ql.querysetstate(int(state_df.iloc[0]))

        trades_df = pd.DataFrame(index=prices_norm.index)
        trades_df[symbol] = 0.0

        daily_returns = prices_norm.pct_change(1)

        position = 0.0
        for index, row in f.iterrows():
            reward = daily_returns.loc[index]
            action = self.ql.query(int(state_df.loc[index]['state']), reward)
            if action == 1 and position == 0.0:
                # go long
                position = self.numLong
                trades_df[symbol][index] = self.numLong
            elif action == 2 and position == 0.0:
                position = -1.0*self.numShort
                trades_df[symbol][index] = -1.0*self.numShort
            elif action == 1 and position < 0.0:
                position = self.numLong
                trades_df[symbol][index] = 2.0*self.numLong
            elif action == 2 and position > 0.0:
                position = -1.0*self.numShort
                trades_df[symbol][index] = -2.0*self.numShort
            elif action == 0:
                trades_df[symbol][index] = 0.0

        return trades_df, benchmark, prices_norm


    def discretize(self, features):
        discrete_df = pd.DataFrame(index=features.index)
        bins = pd.DataFrame(columns=features.columns)

        for (featureName, featureData) in features.iteritems():
           discrete_df[featureName], bins[featureName] = pd.qcut(featureData, q=10, labels=False, retbins=True)

        return discrete_df, bins

    def discretize_test(self, features):
        discrete_df = pd.DataFrame(index=features.index)
        for (featureName, featureData) in features.iteritems():
           discrete_df[featureName] = pd.cut(featureData, bins=self.bins[featureName], labels=False)

        return discrete_df

    def getState(self, discrete_features):
        state_df = pd.DataFrame(index=discrete_features.index, columns=['state'])
        discrete_features.fillna(value=0, inplace=True)
        discrete_features = discrete_features.astype(int)

        for index, row in discrete_features.iterrows():
            uniq_string = ''
            for data in row:
                uniq_string += str(data)
            state_df['state'][index] = uniq_string

        # state_df['state'] = discrete_indicators['macd'].astype(str) \
        #             + discrete_indicators['trix'].astype(str) \
        #             + discrete_indicators['bb value'].astype(str)
        return state_df
