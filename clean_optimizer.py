import funcs
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

# TODO | (1) --------------Make the implementation here match the implementation funcs------------------------
# TODO |     The function within funcs still uses look_back as a parameter, however I still feel doing away
# TODO |         with such a thing would be for the best, instead using two separate, sensible lookbacks (vol vs cov)
# TODO |     Indeed, below I raise an error if any of the price series have less days than the hardcoded cov look back
# TODO |     Either way, the function in funcs (or I suppose, here, but don't recommend it) needs to change to match
# TODO |         each other

# TODO | (2) --------------Investigate (see notes on phone) a better way to forecast covariance------------------------
# TODO |     Steve recommended to us a few separate things to try to find a forecaster that has some predictability
# TODO |     We had tried (backtest.py) to find a look back for covariance that could provide any insight into forward
# TODO |          30 day covariance, however, from our personal back testing, there wasn't any day that provided any
# TODO |          predictability
# TODO |     So we need to new strategies to find a useful way to forecast cov, or instead opt to find a way to predict
# TODO |          bounds on forward covariance, then fit ourselves reasonably within that for each monthly prediction

LOOK_BACK = 90
# all below ETFs use 1/PE_ratio to find annual return, bond ETFs I use actual bond yields
# equal to the average maturity of the holdings of that ETF
SPY_return = 0.0257
TLT_return = 0.0213  # updated using 2/26/2021 bond yields avg of 20 yr and 30 yr yields
EWJ_return = 0.0497
TAN_return = 0.0068
IEF_return = 0.0130  # updated using 2/26/2021 bond yields avg of 7 yr and 10 yr yields
XHE_return = 0.0207


csv_data = pd.read_csv('Excel Sheets/SPY.csv')
spy_adj_close = csv_data["Adj Close"].to_numpy()

csv_data = pd.read_csv('Excel Sheets/TLT.csv')
tlt_adj_close = csv_data["Adj Close"].to_numpy()

csv_data = pd.read_csv('Excel Sheets/EWJ.csv')
ewj_adj_close = csv_data["Adj Close"].to_numpy()

csv_data = pd.read_csv('Excel Sheets/TAN.csv')
tan_adj_close = csv_data["Adj Close"].to_numpy()

csv_data = pd.read_csv('Excel Sheets/IEF.csv')
ief_adj_close = csv_data["Adj Close"].to_numpy()

csv_data = pd.read_csv('Excel Sheets/XHE.csv')
xhe_adj_close = csv_data["Adj Close"].to_numpy()

# ---------------- PARAMETERS -------------------
asset_classes = [
    {'ticker': "SPY", 'annual_return': SPY_return, 'price_list': spy_adj_close},
    {'ticker': "TLT", 'annual_return': TLT_return, 'price_list': tlt_adj_close},
    {'ticker': "EWJ", 'annual_return': EWJ_return, 'price_list': ewj_adj_close},
    {'ticker': "TAN", 'annual_return': TAN_return, 'price_list': tan_adj_close},
    {'ticker': "IEF", 'annual_return': IEF_return, 'price_list': ief_adj_close},
    {'ticker': "XHE", 'annual_return': XHE_return, 'price_list': xhe_adj_close},
]
#
#asset_classes = [
#    {'ticker': "SPY", 'annual_return': SPY_return, 'price_list': spy_adj_close},
#    {'ticker': "TLT", 'annual_return': TLT_return, 'price_list': tlt_adj_close}
#]
vol_look_back = 90  # auto-set (study should be done to alter this number)
cov_look_back = 252  # auto-set (study should be done to alter this number)
graph_results = False  # auto-set
print_results = True  # auto-set
leverage_increment = 0.01  # auto-set
cov_matrix = -1  # auto-set to -1, let parameter override if don't want to calc covariance matrix
iteration_count = 250000  # auto-set at 1m but let this parameter be overrode
annual_contribution = 0  # auto-set to 0 meaning we don't introduce outside cash
# -----------------------------------------------

# the "start point" of the function, given ^ parameters

if cov_matrix == -1:
    calc_cov_matrix = True
else:
    calc_cov_matrix = False

if calc_cov_matrix:
    # calculates minimum look-back as lesser of vol and cov look-backs, then verifies all price lists are at least
    # of an equal length to the minimum look-back
    if vol_look_back > cov_look_back:
        minimum_look_back = cov_look_back
    else:
        minimum_look_back = vol_look_back

    # verifies that all price lists are at least the length of the minimum look-back
    for asset_class in asset_classes:
        if (len(asset_class['price_list']) < minimum_look_back) and (minimum_look_back == cov_look_back):
            raise ValueError("The price list of " + str(asset_class['ticker'])
                             + " is shorter than the look back necessary to calculate covariance " + "("
                             + str(len(asset_class['price_list'])) + " < " + str(cov_look_back) + ").")
        elif (len(asset_class['price_list']) < minimum_look_back) and (minimum_look_back == vol_look_back):
            raise ValueError("The price list of " + str(asset_class['ticker'])
                             + " is shorter than the look back necessary to calculate variance " + "("
                             + str(len(asset_class['price_list'])) + " < " + str(vol_look_back) + ").")

    # creates an n x m array (n = number of asset classes, m = look back); rows are assets, columns are daily prices
    price_list_array = np.zeros(shape=(len(asset_classes), vol_look_back - 1))
    for i in range(len(asset_classes)):
        price_list_array[i] = funcs.price_series_to_return_series(asset_classes[i]['price_list'][-vol_look_back:])

    # creates two price list arrays with row length
    # equal to the parameter 'vol_look_back' and 'cov_look_back' respectively
    # Note: each look-back requires + 1 daily prices as we convert daily prices to daily percent changes
    vol_look_back_price_list_array = price_list_array.copy()
    for i in range(len(vol_look_back_price_list_array)):
        vol_look_back_price_list_array[i] = vol_look_back_price_list_array[i][:vol_look_back]

    cov_look_back_price_list_array = price_list_array.copy()
    for i in range(len(cov_look_back_price_list_array)):
        cov_look_back_price_list_array[i] = cov_look_back_price_list_array[i][:cov_look_back]

    # creates one covariance matrix using different look-backs for volatility and covariance respectively
    # https://quant.stackexchange.com/questions/40102/negative-variance
    # NOTE: Our covariance matrix is created using daily price series, hence we multiply by trading days per yr
    corr_matrix = np.corrcoef(cov_look_back_price_list_array)
    vol_look_back_cov_matrix = np.cov(vol_look_back_price_list_array)
    vol_matrix = np.zeros(shape=(len(asset_classes), len(asset_classes)))
    for i in range(len(asset_classes)):
        vol_matrix[i][i] = np.sqrt(vol_look_back_cov_matrix[i][i])

    cov_matrix = funcs.TRADING_DAYS_PER_YEAR*np.matmul(np.matmul(vol_matrix, corr_matrix), vol_matrix)
    print(cov_matrix)

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

    # check for negative portfolio volatility, which should never be possible
    if np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)) < 0:
        print('----')
        print('NEGATIVE VARIANCE FOUND')
        print('weights: ' + str(rand_weights))
        print('covariance matrix: ' + str(cov_matrix))
        exit()

    portfolio_vol = np.sqrt(np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)))

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
