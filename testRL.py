import myLearner as ml
import XGBoostLearner as xgbl
import ReinforcementLearner as rl
import datetime as dt
import marketsimcode
import features

import pandas as pd
from matplotlib import pyplot as plt

rl = rl.ReinforcementLearner()

symbol= 'SPY'
start_date=dt.datetime(2020,1,1)
end_date=dt.datetime(2020,5,26)

rl.addEvidence(symbol, start_date, end_date)

start_date=dt.datetime(2020,1,1)
end_date=dt.datetime(2020,5,26)

# trades_df, benchmark_df, prices_norm = rl.testPolicy(symbol, start_date, end_date)

N= 10
action_today = pd.DataFrame(index=range(0,N) ,columns=['action', 'prob'])
look_back = 1
for i in range(0,N):
    trades_df, benchmark_df, prices_norm = rl.testPolicy(symbol, start_date, end_date)
    last_few = trades_df.iloc[-1*look_back:,0]
    action = 0.0
    for k in range(look_back-1,-1,-1):
        if last_few[k] == 2000.00 or last_few[k] == -2000.00:
            action = last_few[k]
            break
        else:
            action = last_few[k]

    action_today['action'][i] = action

print(action_today['action'].value_counts())

real_trades_df = trades_df[symbol]
opt_port = marketsimcode.compute_portvals(real_trades_df, symbol)
opt_port = opt_port.divide(opt_port.iloc[0])
benchmark_port = marketsimcode.compute_portvals(benchmark_df, symbol)
benchmark_port = benchmark_port.divide(benchmark_port.iloc[0])

long_dates = trades_df[trades_df[symbol] > 0].index
short_dates = trades_df[trades_df[symbol] < 0].index

plot_window = dt.date(2020,1,1)
plt.plot(opt_port, color='blue')
plt.plot(benchmark_port, color='black')
plt.plot(prices_norm, color='orange')
plt.xlim([plot_window, end_date])
# plt.plot(trades_df['prob'], color='cyan')
plt.grid(True)
plt.xlabel("Date")
plt.ylabel("Normalized Value")
for date in long_dates:
    plt.axvline(x=date, color='green')
for date in short_dates:
    plt.axvline(x=date, color='red')

plt.title(symbol + " Benchmark Normalized Values")
plt.legend(['Optimal', 'Benchmark', 'Price'])
plt.gcf().autofmt_xdate()
plt.show()
