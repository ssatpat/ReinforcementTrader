import pandas as pd
import numpy as np  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import datetime as dt  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
import os  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_intraday
from iexfinance.stocks import get_historical_data

def author():
    return 'ssatpathy9'

def compute_portvals(order, symbol, start_val = 100000, commission=0.0, impact=0.0):
    orders = pd.DataFrame()
    # orders = orders.rename(columns={symbol: 'Shares'})
    orders['Shares'] = order
    orders['Symbol'] = symbol
    orders['Order'] = np.where(orders['Shares'] <= 0, 'SELL', 'BUY')
    orders['Shares'] = np.where(orders['Shares'] <= 0, -1.0 * orders['Shares'], orders['Shares'])
    orders['Order'] = np.where(orders['Shares'] == 0, 'HOLD', orders['Order'])

    orders_df = orders

    #Get start and end date of order book
    start_date = orders_df.index[0]
    end_date = orders_df.index[-1]

    #Get all symbols from orders file
    # symbols = orders_df["Symbol"].unique()
    # symbols = ["JPM"]
    #Construct prices dataframe and add CASH column
    # prices = get_data(symbols, pd.date_range(start_date, end_date))
    data = get_historical_data(symbol, start=start_date, end=end_date, output_format='pandas')
    # data = get_historical_intraday(symbol, date=end_date, output_format='pandas')
    prices = data['close'].to_frame(symbol)
    # prices = prices.drop(columns=['SPY'], axis=1)
    prices['CASH']= np.ones((len(prices),1))
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='backfill', inplace=True)

    #Build trades dataframe and initialize with zeros
    trades = prices.copy(deep=True)
    for col in trades.columns:
        trades[col].values[:] = 0.0

    for index, row in orders_df.iterrows():
        if row['Order'].lower() == "buy":
            trades.loc[index][row['Symbol']] = row['Shares'] + trades.loc[index][row['Symbol']]
            cash_value = -1.0 * prices.loc[index][row['Symbol']] * row['Shares'] * (1+impact) - commission
            trades.loc[index]['CASH'] = trades.loc[index]['CASH'] + cash_value

        if row['Order'].lower() == "sell":
            trades.loc[index][row['Symbol']] = trades.loc[index][row['Symbol']] - row['Shares']
            cash_value = prices.loc[index][row['Symbol']] * row['Shares'] * (1-impact) - commission
            trades.loc[index]['CASH'] = trades.loc[index]['CASH'] + cash_value



    #build holdings dataframe from trades and initialize with zeros
    holdings = trades.copy(deep=True)
    for col in holdings.columns:
        holdings[col].values[:] = 0.0

    #special case for first row
    holdings.loc[trades.index[0]] = trades.loc[trades.index[0]]
    holdings.loc[trades.index[0]]['CASH'] = start_val + trades.loc[trades.index[0]]['CASH']

    #iterate over holdings starting from second row
    for i in range(1, holdings.shape[0]):
        for j in range(holdings.shape[1]):
            holdings.loc[holdings.index[i],holdings.columns[j]] = holdings.loc[holdings.index[i-1],holdings.columns[j]] \
                                                                  + trades.loc[holdings.index[i],holdings.columns[j]]
    #create values dataframe
    #values = holdings*prices
    values = holdings.copy(deep=True)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            values.loc[values.index[i],values.columns[j]] = values.loc[values.index[i],values.columns[j]]*prices.loc[values.index[i],values.columns[j]]


    # for index, row in values.iterrows():
    #     if index in prices.index:
    #         for i in symbols:
    #             values.loc[index][i] = values.loc[index][i]*prices.loc[index][i]
    #             prev_index = index
    #     else:
    #         for i in symbols:
    #             values.loc[index][i] = values.loc[prev_index][i]
    #             prev_index = index

    portvals = values.sum(axis=1)

    return portvals
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
def test_code():  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # this is a helper function you can use to test your code  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # note that during autograding his function will not be called.  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Define input parameters  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    of = "C:\\Users\\ssatpathy\\Documents\\OMSCS GATech\\ML4T\\ML4T_2020Spring\\marketsim\\orders\\orders-05.csv"
    sv = 1000000
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Process orders  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    portvals = compute_portvals(orders_file = of, start_val = sv)  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    if isinstance(portvals, pd.DataFrame):  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        portvals = portvals[portvals.columns[0]] # just get the first column  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    else:  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
        "warning, code did not return a DataFrame"  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Get portfolio stats  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cum_ret}")
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")
    print()
    print(f"Standard Deviation of Fund: {std_daily_ret}")
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")
    print()
    print(f"Average Daily Return of Fund: {avg_daily_ret}")
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")
  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
if __name__ == "__main__":  		  	   		     			  		 			     			  	  		 	  	 		 			  		  			
    print("lol")
