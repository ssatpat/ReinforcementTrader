from indicators import *
import pandas as pd
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_intraday
from iexfinance.stocks import get_historical_data
import matplotlib.pyplot as plt

IEX_TOKEN='Tsk_ed36ac7a3cf94e53a943ab13258876a2'
IEX_API_VERSION='iexcloud-sandbox'
def generateFeatures(symbol, start_date, end_date):

    data = get_historical_data(symbol, start=start_date, end=end_date, output_format='pandas')
    # data = get_historical_intraday(symbol, date=end_date, output_format='pandas')
    prices = data['close'].to_frame(symbol)
    # volume_df = data['volume'].to_frame('volume')
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='backfill', inplace=True)

    prices_norm = prices.divide(prices.iloc[0])
    bollinger_df = do_bollinger(prices_norm, 20)
    macd_df = do_macd(prices_norm)
    trix_df = do_trix(prices_norm,14)

    final_df = pd.concat([bollinger_df, macd_df, trix_df, data], axis=1)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    return final_df

def discretize(indicators):
    indicators.dropna()
    discrete_df = pd.DataFrame(index=indicators.index, columns=indicators.columns)
    for column, data in indicators.iteritems():
        discrete_df[column] = pd.qcut(data, q=10, labels=False)
    return discrete_df

if __name__ == "__main__":
    symbol = 'AAPL'
    sd =dt.datetime(2019, 4, 1)
    ed = dt.datetime(2020, 4, 21)
    features = generateFeatures(symbol, sd, ed)
    discretized_features = discretize(features)



    plt.plot(discretized_features['bb value'])
    plt.show()



