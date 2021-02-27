import math
import funcs
import pandas as pd
import numpy as np
import sys

# 11/9/2020

# TODO ~ GOAL: use monte-carlo to brute force an estimate for optimal weights for n asset classes
# TODO (1) Current runs having optimal portfolio volatility at 80% which shouldn't be possible
# two solutions to this; create your own covariance matrix OR make funcs function to convert adj_close price list
# into a list of one less length of ln(i+1/i) daily returns, then run it again
# TODO (2) If you want to use all 3 (spy, tlt, ewj) again, ctrl+f all the 'debug' stuff
# TODO **(3) I'd very much like this process turned into a function by which we feed in a list of dicts
# with like, adj_close list, ticker name (tlt, ewj, spy), and we get top 5 portfolio's and optimal leverages back
# TODO (4) Create function which takes optimal portfolio and employs it on $10,000 on a random distribution
# or hmm, maybe won't be legit
# TODO **************TAKE TESTER2 AND MAKE IT A FUNCTION IN FUNCS! IT WORKS EXCELLENTLY**************
# TODO ********USE ABOVE FUNCTION (WHEN U PORT IT TO FUNCS) TO USE P/E RATIO EXCEL FROM QUANDL TO DO RANDOM TESTS
# above will require, in addition, SPY adj close and TLT adj close with long time horizon

# 90 look back seems fair?

LOOK_BACK = 90

csv_data = pd.read_csv('Excel Sheets/SPY_current.csv')
spy_adj_close = csv_data["Adj Close"].to_numpy()
spy_rv = funcs.calc_annualized_realized_vol(spy_adj_close, LOOK_BACK)
spy_mean_return = funcs.calc_annualized_arithmetic_mean(spy_adj_close, LOOK_BACK)
spy_mean_geometric_return = funcs.calc_annualized_geometric_mean(spy_adj_close, LOOK_BACK)

csv_data = pd.read_csv('Excel Sheets/TLT_current.csv')
tlt_adj_close = csv_data["Adj Close"].to_numpy()
tlt_rv = funcs.calc_annualized_realized_vol(tlt_adj_close, LOOK_BACK)
tlt_mean_return = funcs.calc_annualized_arithmetic_mean(tlt_adj_close, LOOK_BACK)
tlt_mean_geometric_return = funcs.calc_annualized_geometric_mean(tlt_adj_close, LOOK_BACK)

csv_data = pd.read_csv('Excel Sheets/EWJ_current.csv')
ewj_adj_close = csv_data["Adj Close"].to_numpy()
ewj_rv = funcs.calc_annualized_realized_vol(ewj_adj_close, LOOK_BACK)
ewj_mean_return = funcs.calc_annualized_arithmetic_mean(ewj_adj_close, LOOK_BACK)
ewj_mean_geometric_return = funcs.calc_annualized_geometric_mean(ewj_adj_close, LOOK_BACK)

# SPY and EWJ I calculate earnings yield, BONDs just straight rip 30 year treasury bill yield
SPY_return = 0.0277  # https://www.multpl.com/s-p-500-earnings-yield
TLT_return = 0.0163  # https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield
EWJ_return = 0.058  # https://www.ishares.com/us/products/239665/ishares-msci-japan-etf for the P/E ratio
# https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf -> Average Yield to Maturity

# DEBUG
'''
adj_close_matrix = np.array([
    funcs.price_series_to_return_series(spy_adj_close[-LOOK_BACK:]),
    funcs.price_series_to_return_series(tlt_adj_close[-LOOK_BACK:]),
    funcs.price_series_to_return_series(ewj_adj_close[-LOOK_BACK:])
])
mean_returns = [SPY_return, TLT_return, EWJ_return]
'''

adj_close_matrix = np.array([
    funcs.price_series_to_return_series(spy_adj_close[-LOOK_BACK:]),
    funcs.price_series_to_return_series(tlt_adj_close[-LOOK_BACK:])
])
mean_returns = [SPY_return, TLT_return]
cov_matrix = np.cov(adj_close_matrix)
print(cov_matrix)

print("90-day look back SPY rv: " + str(spy_rv) + " | Ripped SPY Annualized Return: " +
      str(SPY_return))
print("90-day look back BOND rv: " + str(tlt_rv) + " | Ripped BOND Annualized Return: " +
      str(TLT_return))
print("90-day look back EWJ rv: " + str(ewj_rv) + " | Calc'd EWJ Annualized Return: " +
      str(EWJ_return))

sharpe_ratios = []  # dicts should have format {"sharpe_ratio": x, "stock_proportion": y, "bond_proportion": z}
iteration_count = 1000000
for i in range(iteration_count):
    rand_weights = np.random.random(len(adj_close_matrix))
    rand_weights /= np.sum(rand_weights)

    portfolio_return = 0
    for j in range(len(rand_weights)):
        portfolio_return += rand_weights[j] * mean_returns[j]

    portfolio_vol = np.sqrt(252)*np.sqrt(np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)))

    # can use BOND_return comment link -> 10 year bonds as risk free rate if doing sharpe (mean - risk_free) / vol
    sharpe_ratio = portfolio_return / portfolio_vol
    sharpe_ratios.append({"sharpe_ratio": sharpe_ratio, "spy_proportion": rand_weights[0],
                          "tlt_proportion": rand_weights[1], 'portfolio_return': portfolio_return,
                          'portfolio_vol': portfolio_vol})

    # DEBUG
    '''
    sharpe_ratios.append({"sharpe_ratio": sharpe_ratio, "spy_proportion": rand_weights[0],
                          "tlt_proportion": rand_weights[1], "ewj_proportion": rand_weights[2],
                          'portfolio_return': portfolio_return, 'portfolio_vol': portfolio_vol})
    '''
print(len(sharpe_ratios))
sharpe_ratios = sorted(sharpe_ratios, key=lambda i: i['sharpe_ratio'])
print(sharpe_ratios[-5:])

best_portfolio = sharpe_ratios[-1]

leverage = 0
leverage_increment = 0.01
last_geo_mean = -sys.maxsize
while True:
    leverage += leverage_increment
    levered_return = leverage*best_portfolio['portfolio_return']
    levered_vol = leverage*best_portfolio['portfolio_vol']

    current_geo_mean = levered_return - ((levered_vol**2)/2)
    if current_geo_mean < last_geo_mean:
        break

    last_geo_mean = current_geo_mean

print("optimal leverage is " + str(leverage))
print("for a return of " + str(leverage*best_portfolio['portfolio_return']) + " | and volatility of " +
      str(leverage*best_portfolio['portfolio_vol']))
