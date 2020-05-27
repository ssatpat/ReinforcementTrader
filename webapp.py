from flask import Flask
from flask import render_template
from flask import request
from flask import send_file

from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.backends.backend_svg import FigureCanvasSVG

from matplotlib.figure import Figure
import io
import random
import base64
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import matplotlib.cbook as cbook


# import myLearner as ml
# import XGBoostLearner as xgbl
import ReinforcementLearner as rl
import datetime as dt
import marketsimcode
import features

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('analysis.html')

@app.route('/submit', methods=['POST'])
def onSubmit():
    symbol = request.form.get('stock')
    start_date = request.form.get('startDate')
    end_date = request.form.get('endDate')
    port_start_val = int(request.form.get('startVal'))
    numLong = request.form.get('numLong')
    numShort = request.form.get('numShort')

    l = rl.ReinforcementLearner(numLong=float(numLong),numShort=float(numShort))
    symbol = symbol
    # sd = dt.datetime(2015, 1, 1)
    train_start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    # train_start_date = sd
    train_end_date = dt.datetime.strptime(end_date,"%Y-%m-%d")

    l.addEvidence(symbol, train_start_date, train_end_date)

    start_date = dt.datetime.strptime(start_date, "%Y-%m-%d")
    end_date = dt.datetime.strptime(end_date,"%Y-%m-%d")

    trades_df, benchmark_df, prices_norm = l.testPolicy(symbol, start_date, end_date)
    real_trades_df = trades_df[symbol]
    opt_port = marketsimcode.compute_portvals(real_trades_df, symbol,start_val=port_start_val)
    opt_port = opt_port.divide(opt_port.iloc[0])
    benchmark_port = marketsimcode.compute_portvals(benchmark_df, symbol, start_val=port_start_val)
    benchmark_port = benchmark_port.divide(benchmark_port.iloc[0])


    long_dates = trades_df[trades_df[symbol] > 0].index
    short_dates = trades_df[trades_df[symbol] < 0].index

    fig = create_figure(opt_port, benchmark_port, long_dates, short_dates, prices_norm)
    output = io.BytesIO()
    output.seek(0)
    FigureCanvas(fig).print_png(output)
    img = base64.b64encode((output.getvalue()))

    return render_template('image.html', name=symbol, image_data=img.decode('ascii'))

@app.route('/plot.png')
def plot():
    return render_template('image.html', name = 'ShreyTest', url ='/plot.png')

# def create_figure(opt_port, benchmark_port, long_dates, short_dates):
#     fig = Figure()
#     axis = fig.add_subplot(1, 1, 1)
#     xs = range(100)
#     ys = [random.randint(1, 50) for x in xs]
#     axis.plot(xs, ys)
#     return fig


# @app.route('/plot.png')
# def plot_png(opt_port, benchmark_port, long_dates, short_dates):
#     fig = create_figure(opt_port, benchmark_port, long_dates, short_dates)
#     output = io.BytesIO()
#     FigureCanvasSVG(fig).print_svg(output)
#     Response(output.getvalue(), mimetype='image/svg+xml')
#     '''< img
#     src = "/plot.png"
#     alt = "my plot" >'''
#     return render_template("hello.html")

def create_figure(opt_port, benchmark_port, long_dates, short_dates, prices_norm):
    fig = Figure(figsize=(12, 6))
    axis = fig.add_subplot(1, 1, 1)

    axis.plot(opt_port, color='blue')
    axis.plot(benchmark_port, color='black')
    axis.plot(prices_norm, color='orange')
    axis.grid(True)
    axis.legend(['Optimal', 'Benchmark'])
    axis.set_xlabel("Date")
    axis.set_ylabel("Normalized Value")
    for date in long_dates:
        axis.axvline(x=date, color='green')
    for date in short_dates:
        axis.axvline(x=date, color='red')

    axis.set_title("SYM" + " Benchmark Normalized Values")

    axis.grid(True)
    fig.autofmt_xdate()

    # url = '/static/images/plot.png'
    # fig.savefig('/static/images/plot.png')

    return fig
