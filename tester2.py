import math
import funcs
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# TODO | (1) ----------------Solve negative vols resulting from separate look backs------------------
# TODO |     This function does away with a given look_back parameter and attempts to implement a vol_look_back
# TODO |          and cov_look_back, as optimal look back for vol is different than that of covariance
# TODO |     The issue, however, is that creating a covariance matrix with different look backs results in a
# TODO |         negative portfolio volatilities for certain weights
# TODO |     To solve this, I currently recommend implementing a back-testing function which could test
# TODO |         which singular look back finds the covariances matrices which are best predictor of forward covariance
# TODO |         matrices.
# TODO |     Otherwise, can find portfolio vol with both look backs, and blend those two numbers
# TODO |     **STEVE MAINTAINS THAT A NEGATIVE RESULT SHOULDN'T BE POSSIBLE STILL (requires algebraic investigation)

# TODO | (2) --------------Make the implementation here match the implementation funcs------------------------
# TODO |     The function within funcs still uses look_back as a parameter, however I still feel doing away
# TODO |         with such a thing would be for the best, instead using two separate, sensible lookbacks (vol vs cov)
# TODO |     Indeed, below I raise an error if any of the price series have less days than the hardcoded cov look back
# TODO |     Either way, the function in funcs (or I suppose, here, but don't recommend it) needs to change to match
# TODO |         each other

# TODO | (3) --------------Investigate (see notes on phone) a better way to forecast covariance------------------------
# TODO |     Steve recommended to us a few separate things to try to find a forecaster that has some predictability
# TODO |     We had tried (backtest.py) to find a look back for covariance that could provide any insight into forward
# TODO |          30 day covariance, however, from our personal back testing, there wasn't any day that provided any
# TODO |          predictability
# TODO |     So we need to new strategies to find a useful way to forecast cov, or instead opt to find a way to predict
# TODO |          bounds on forward covariance, then fit ourselves reasonably within that for each monthly prediction

LOOK_BACK = 90
SPY_return = 0.0277
TLT_return = 0.0163

csv_data = pd.read_csv('Excel Sheets/SPY_current.csv')
spy_adj_close = csv_data["Adj Close"].to_numpy()

csv_data = pd.read_csv('Excel Sheets/TLT_current.csv')
tlt_adj_close = csv_data["Adj Close"].to_numpy()

# ---------------- PARAMETERS -------------------
asset_classes = [
    {'ticker': "SPY", 'annual_return': 0.0277, 'price_list': spy_adj_close},
    {'ticker': "TLT", 'annual_return': 0.0163, 'price_list': tlt_adj_close}
]
vol_look_back = 90  # auto-set (study should be done to alter this number)
cov_look_back = 252  # auto-set (study should be done to alter this number)
graph_results = True  # auto-set
print_results = True  # auto-set
leverage_increment = 0.01  # auto-set
cov_matrix = -1  # auto-set to -1, let parameter override if don't want to calc covariance matrix
iteration_count = 100000  # auto-set at 1m but let this parameter be overrode
annual_contribution = 0  # auto-set to 0 meaning we don't introduce outside cash
# -----------------------------------------------

# the "start point" of the function, given ^ parameters

if cov_matrix == -1:
    calc_cov_matrix = True
else:
    calc_cov_matrix = False

if calc_cov_matrix:
    for asset_class in asset_classes:
        if len(asset_class['price_list']) < cov_look_back:
            raise ValueError("The price list of " + str(asset_class['ticker'])
                             + " is shorter than the look back necessary to calculate covariance " + "("
                             + str(len(asset_class['price_list'])) + " < " + str(cov_look_back) + ").")

    # creates an n x m array (n = number of asset classes, m = look back); rows are assets, columns are daily prices
    price_list_array = np.zeros(shape=(len(asset_classes), vol_look_back - 1))
    for i in range(len(asset_classes)):
        price_list_array[i] = funcs.price_series_to_return_series(asset_classes[i]['price_list'][-vol_look_back:])
    vol_look_back_cov_matrix = 252*np.cov(price_list_array)  # DELETE THE 252
    print(vol_look_back_cov_matrix)

    price_list_array = np.zeros(shape=(len(asset_classes), cov_look_back - 1))
    for i in range(len(asset_classes)):
        price_list_array[i] = funcs.price_series_to_return_series(asset_classes[i]['price_list'][-cov_look_back:])

    cov_look_back_cov_matrix = 252*np.cov(price_list_array)  # DELETE THE 252
    print(cov_look_back_cov_matrix)

    # our overall covariance matrix will have variances (diagonal entries)
    # with a different look back than for covariances, requiring above, and directly below this
    cov_matrix = cov_look_back_cov_matrix.copy()
    for i in range(len(cov_matrix)):
        cov_matrix[i][i] = vol_look_back_cov_matrix.copy()[i][i]

    print('----') # DELETE v-- use the below; dont delete it
    corr_matrix = np.corrcoef(price_list_array)
    print(corr_matrix)
    vol_matrix = np.zeros(shape=(len(asset_classes), len(asset_classes)))
    for i in range(len(asset_classes)):
        vol_matrix[i][i] = np.sqrt(cov_matrix[i][i])

    print(np.matmul(np.matmul(vol_matrix, corr_matrix), vol_matrix))
    print('----')

    # print(math.sqrt(252)*math.sqrt(vol_look_back_cov_matrix[0][0])) DELETE RE-PRINT THESE
    # print(math.sqrt(252)*math.sqrt(cov_look_back_cov_matrix[0][0])) DELETE RE-PRINT THESE
    print(math.sqrt(vol_look_back_cov_matrix[0][0]))
    print(math.sqrt(cov_look_back_cov_matrix[0][0]))
    print(cov_matrix)
    cov_matrix = np.matmul(np.matmul(vol_matrix, corr_matrix), vol_matrix) # DELETE this solves it o_O think on it


# initialize list of mean annual returns in proper order
mean_returns = []
for asset_class in asset_classes:
    mean_returns.append(asset_class['annual_return'])

portfolios = []
for i in range(iteration_count):
    rand_weights = np.random.random(len(asset_classes))
    rand_weights /= np.sum(rand_weights)

    portfolio_return = 0
    for j in range(len(rand_weights)):
        portfolio_return += rand_weights[j] * mean_returns[j]

    # print(rand_weights)  # DELETE THIS PRINT STATEMENT
    # portfolio_vol = np.sqrt(252) * np.sqrt(np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)))
    if np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)) < 0:
        print('----')
        print(rand_weights)
        print(cov_matrix)
        exit()

    portfolio_vol = np.sqrt(np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)))  # DELETE (USE ABOVE STATEMENT)

    # can use BOND_return comment link -> 10 year bonds as risk free rate if doing sharpe (mean - risk_free) / vol
    sharpe_ratio = portfolio_return / portfolio_vol
    portfolios.append({'sharpe_ratio': sharpe_ratio, 'portfolio_return': portfolio_return,
                       'portfolio_vol': portfolio_vol, 'weights': rand_weights})

portfolios = sorted(portfolios, key=lambda i: i['sharpe_ratio'])

best_portfolio = portfolios[-1]

leverage = 0
leverages = []  # used in graphing
geometric_means = []  # used in graphing
last_geo_mean = -sys.maxsize
iteration_cap = 10000000
current_iteration = 0
while True:
    current_iteration += 1
    leverage += leverage_increment
    levered_return = leverage * best_portfolio['portfolio_return']
    levered_vol = leverage * best_portfolio['portfolio_vol']

    current_geo_mean = levered_return - ((levered_vol ** 2) / 2)
    leverages.append(leverage)  # used in graphing
    geometric_means.append(current_geo_mean)  # used in graphing
    if current_geo_mean < last_geo_mean:
        break
    if current_iteration < iteration_cap:
        ValueError("For some reason, while calculating the optimal leverage to provide the largest geometric mean, "
                   "there were more than " + str(iteration_cap) + " iterations, which probably shouldn't occur.")

    last_geo_mean = current_geo_mean

optimal_leverage = leverage

portfolios_to_print = 5
if print_results:
    column_list = ["Return", "Vol", "Leveraged Return", "Leveraged Vol"]
    for asset_class in asset_classes:
        column_list.append(asset_class['ticker'])

    # 4 comes from len of column_list before it gets appended any items
    results_array = np.zeros(shape=(portfolios_to_print, 4 + len(asset_classes)))
    for i in range(0, portfolios_to_print):
        current_portfolio = portfolios[i - portfolios_to_print]
        current_row = [current_portfolio['portfolio_return'], current_portfolio['portfolio_vol'],
                       optimal_leverage * current_portfolio['portfolio_return'],
                       optimal_leverage * current_portfolio['portfolio_vol']]
        for j in range(0, len(current_portfolio['weights'])):
            current_row.append(current_portfolio['weights'][j])

        results_array[i] = current_row

    print("The maximum geometric mean is " + str(last_geo_mean) + ", and is obtained with a leverage of "
          + str(optimal_leverage))
    print(pd.DataFrame(results_array, columns=column_list))

print("5 worst portfolios by sharpe")
print(portfolios[:5])
print("5 best portfolios by sharpe")
print(portfolios[-5:])

# have leverages, geometric_means lists to aid in graphing
if graph_results:
    # PRINTING WEIGHTS VS SHARPE (WILL NEED TO BE MORE CREATIVE WITH MORE THAN TWO ASSETS)
    stock_weights = []
    sharpe_ratios = []
    for portfolio in portfolios:
        stock_weights.append(portfolio["weights"][0])
        sharpe_ratios.append(portfolio["sharpe_ratio"])

    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Stock Proportion")
    plt.scatter(stock_weights, sharpe_ratios)
    plt.show()

    # PRINTING LEVERAGE VS GEOMETRIC MEAN
    plt.ylabel("Geometric Mean")
    plt.xlabel("Leverage")
    plt.plot(leverages, geometric_means)
    plt.show()
