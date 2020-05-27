#file containing the code for technical indicators
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_intraday
from iexfinance.stocks import get_historical_data
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

def author():
    return 'ssatpathy9'

def do_technical_analysis(symbol= ['JPM'], start_date=dt.datetime(2008,1,1), end_date=dt.datetime(2009,12,31)):
    data = get_historical_data(symbol, start=start_date, end=end_date, output_format='pandas')
    #data = get_historical_intraday(symbol, date=end_date, output_format='pandas')
    prices = data['close'].to_frame(symbol)

    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='backfill', inplace=True)

    prices_SPY = prices['SPY']
    prices = prices.drop(columns=['SPY'], axis=1)


    prices_norm = prices.divide(prices.iloc[0])

    do_sma(prices_norm,10)
    do_bollinger(prices_norm, 20)
    do_momentum(prices_norm, 10)
    do_macd(prices_norm)
    do_trix(prices_norm,14)

    return

def do_sma(prices, window):
    result = prices.copy(deep=True)
    result['SMA'] = result.rolling(window=window).mean()
    result['Price/SMA'] = result['JPM'] / result['SMA']

    plt.plot(result['SMA'])
    plt.plot(result['JPM'])
    plt.plot(result['Price/SMA'])
    plt.grid(True)
    plt.legend(['SMA', 'Price', 'Price/SMA'])
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.title("Figure 1 - Simple Moving Average and Closing Price of JPM")
    plt.gcf().autofmt_xdate()

    plt.savefig('Figure1.png')
    plt.close()


def do_bollinger(prices, window):
    result = prices.copy(deep=True)
    mv_avg = result.rolling(window=window).mean()
    mv_std = result.rolling(window=window).std()

    upper_band = mv_avg + (2*mv_std)
    lower_band = mv_avg - (2*mv_std)
    bb_value = (result - mv_avg)/(2*mv_std)


    result['upper band'] = upper_band
    result['lower band'] = lower_band
    result['bb value'] = bb_value
    result['moving average'] = mv_avg
    return result
    # fig, axs = plt.subplots(2, sharex=True)
    # fig.suptitle('Figure 2 - Bollinger Bands and BB Value for JPM')
    # axs[0].plot(result['upper band'])
    # axs[0].plot(result['JPM'])
    # axs[0].plot(result['lower band'])
    # axs[0].plot(result['moving average'])
    #
    # axs[0].legend(['upper', 'Price', 'lower', 'SMA'])
    #
    # axs[1].plot(result['bb value'])
    # axs[1].legend(['BB Value'])
    # plt.gcf().autofmt_xdate()
    #
    #
    # plt.savefig('Figure2.png')
    # plt.close()


def do_momentum(prices, n):
    mom = prices.divide(prices.shift(n)) - 1
    mom = mom.rename(columns={"JPM": "Momentum"})

    return mom

    # plt.plot(mom["Momentum"])
    # plt.grid(True)
    # plt.legend(['Momentum'])
    # plt.xlabel("Date")
    # plt.ylabel("Normalized Value")
    # plt.title("Figure 3 - Momentum (Rate of Change) of JPM")
    # plt.gcf().autofmt_xdate()
    #
    # plt.savefig('Figure3.png')
    # plt.close()


def do_macd(prices):
    result = prices.copy(deep=True)
    ema1 = prices.ewm(span=12).mean()
    ema2 = prices.ewm(span=26).mean()
    macd = ema1 - ema2
    signal = macd.rolling(window=9).mean()
    result['macd'] = macd
    result['signal'] = signal

    return result

    # plt.plot(macd)
    # plt.plot(signal)
    # plt.grid(True)
    # plt.legend(['MACD', 'Signal'])
    # plt.xlabel("Date")
    # plt.ylabel("Normalized Value")
    # plt.title("Figure 4 - MACD (12,26,9) for JPM ")
    # plt.gcf().autofmt_xdate()
    #
    # plt.savefig('Figure4.png')
    # plt.close()


def do_trix(prices, n):
    result = prices.copy(deep=True)

    ema1 = prices.ewm(span=n).mean()
    ema2 = ema1.ewm(span=n).mean()
    ema3 = ema2.ewm(span=n).mean()

    trix = ema3.divide(ema3.shift(1)) - 1
    result['trix'] = trix

    return result
    # plt.plot(trix)
    # plt.grid(True)
    # plt.legend(['Trix'])
    # plt.xlabel("Date")
    # plt.ylabel("Normalized Value")
    # plt.title("Figure 5 - TRIX(14) for JPM ")
    # plt.gcf().autofmt_xdate()
    #
    # plt.savefig('Figure5.png')
    # plt.close()


if __name__ == "__main__":
    do_technical_analysis()