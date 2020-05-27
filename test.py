from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_intraday
from iexfinance.stocks import get_historical_data
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


def author():
    return 'ssatpathy9'

def do_technical_analysis(symbol= 'SNAP', start_date=dt.datetime(2019,4,1), end_date=dt.datetime(2020,4,22)):

    a = Stock(symbol, output_format='pandas')

    data = get_historical_data(symbol, start=start_date, end=end_date, output_format='pandas')
    #data = get_historical_intraday(symbol, date=end_date, output_format='pandas')
    prices = data['close'].to_frame(symbol)
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='backfill', inplace=True)

    # prices_SPY = prices['SPY']
    # prices = prices.drop(columns=['SPY'], axis=1)


    prices_norm = prices.divide(prices.iloc[0])

    do_sma(prices_norm,10, symbol)
    do_bollinger(prices_norm, 20, symbol)
    do_momentum(prices_norm, 10, symbol)
    do_macd(prices_norm, symbol)
    do_trix(prices_norm,14, symbol)

    return

def do_sma(prices, window, symbol):
    result = prices.copy(deep=True)
    result['SMA'] = result.rolling(window=window).mean()
    result['Price/SMA'] = result[symbol] / result['SMA']

    plt.plot(result['SMA'])
    plt.plot(result[symbol])
    plt.plot(result['Price/SMA'])
    plt.grid(True)
    plt.legend(['SMA', 'Price', 'Price/SMA'])
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.title("Figure 1 - Simple Moving Average and Closing Price of "+ symbol)
    plt.gcf().autofmt_xdate()

    plt.savefig(symbol+'_SMA.png')
    plt.close()


def do_bollinger(prices, window, symbol):
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

    fig, axs = plt.subplots(1, sharex=True)
    fig.suptitle('Figure 2 - Bollinger Bands and BB Value for '+symbol)
    axs.plot(result['upper band'])
    axs.plot(result[symbol])
    axs.plot(result['lower band'])
    axs.plot(result['moving average'])

    axs.legend(['upper', 'Price', 'lower', 'SMA'])

    # axs[1].plot(result['bb value'])
    # axs[1].legend(['BB Value'])
    plt.gcf().autofmt_xdate()


    plt.savefig(symbol+'_Bollinger.png')
    plt.close()


def do_momentum(prices, n, symbol):
    mom = prices.divide(prices.shift(n)) - 1
    mom = mom.rename(columns={symbol: "Momentum"})

    plt.plot(mom["Momentum"])
    plt.grid(True)
    plt.legend(['Momentum'])
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.title("Figure 3 - Momentum (Rate of Change) of "+ symbol)
    plt.gcf().autofmt_xdate()

    plt.savefig(symbol+'_momentum.png')
    plt.close()


def do_macd(prices, symbol):
    ema1 = prices.ewm(span=12).mean()
    ema2 = prices.ewm(span=26).mean()
    macd = ema1 - ema2
    signal = macd.rolling(window=9).mean()

    plt.plot(macd)
    plt.plot(signal)
    plt.grid(True)
    plt.legend(['MACD', 'Signal'])
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.title("Figure 4 - MACD (12,26,9) for "+symbol)
    plt.gcf().autofmt_xdate()

    plt.savefig(symbol+'_MACD.png')
    plt.close()


def do_trix(prices, n, symbol):
    ema1 = prices.ewm(span=n).mean()
    ema2 = ema1.ewm(span=n).mean()
    ema3 = ema2.ewm(span=n).mean()

    trix = ema3.divide(ema3.shift(1)) - 1

    plt.plot(trix)
    plt.grid(True)
    plt.legend(['Trix'])
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.title("Figure 5 - TRIX(14) for "+ symbol)
    plt.gcf().autofmt_xdate()

    plt.savefig(symbol+'_trix.png')
    plt.close()


if __name__ == "__main__":
    do_technical_analysis()