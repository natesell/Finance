import numpy as np
import re
from io import StringIO
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import sys
from alpha_vantage.timeseries import TimeSeries

TRADING_DAYS_PER_YEAR = 252

# Honestly all I can confirm correct so far via testing are:
# calc_annualized_realized_vol(prices, days=-1):
# calc_annualized_arithmetic_mean(prices, days=-1):
# calc_annualized_geometric_mean(prices, look_back=-1):


# Used like 'df = YahooFinanceHistory('AAPL', days_back=365).get_quote()'
# Literally stolen from:
# https://stackoverflow.com/questions/44225771/scraping-historical-data-from-yahoo-finance-with-python
class YahooFinanceHistory:
    timeout = 2
    crumb_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
    crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history&crumb={crumb}'

    def __init__(self, symbol, days_back=7):
        self.symbol = symbol
        self.session = requests.Session()
        self.dt = timedelta(days=days_back)

    def get_crumb(self):
        response = self.session.get(self.crumb_link.format(self.symbol), timeout=self.timeout)
        response.raise_for_status()
        match = re.search(self.crumble_regex, response.text)
        if not match:
            raise ValueError('Could not get crumb from Yahoo Finance')
        else:
            self.crumb = match.group(1)

    def get_quote(self):
        if not hasattr(self, 'crumb') or len(self.session.cookies) == 0:
            self.get_crumb()
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        datefrom = int((now - self.dt).timestamp())
        url = self.quote_link.format(quote=self.symbol, dfrom=datefrom, dto=dateto, crumb=self.crumb)
        response = self.session.get(url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])


class APIQuery:
    # Note: TimeSeries normally takes one input which is your AlphaVantage API key.
    #       If you have your API key saved in an environment variable called 'ALPHAVANTAGE_API_KEY',
    #       then your API key will automatically be pulled. I've excluded my API key for safety.
    ts = TimeSeries(output_format='pandas')

    # "ticker" param is a string of the stock ticker you are querying
    # returns a pandas Series object with full history of daily adjusted price history by date
    @staticmethod
    def get_daily_adj_close_as_series(ticker):
        data, metadata = APIQuery.ts.get_daily_adjusted(ticker, outputsize='full')
        if data.isnull().sum().sum() > 0:
            raise ValueError("There is at least one piece of data that is null.")
        else:
            return data['5. adjusted close']


# asset_classes = [
#    {'ticker': "SPY", 'annual_return': 0.0277, 'price_list': spy_adj_close},
#    {'ticker': "TLT", 'annual_return': 0.0163, 'price_list': tlt_adj_close}
# ]
# look_back = 90 (number of days to look back in price series when calculating variance/covariance)
# graph_results = False (auto-set, can override)
# print_results = True  (auto-set, can override)
# leverage_increment = 0.01  (auto-set, can override)
# cov_matrix = -1  auto-set (can override) to -1, let parameter override if don't want to calc covariance matrix
# iteration_count = 100000  auto-set (can override) at 1m but let this parameter be overrode
# annual_contribution = 0 auto-set at zero, this is the amount of dollars you will invest each year
def portfolio_optimizer(asset_classes, look_back, graph_results=True, print_results=True,
                        leverage_increment=0.01, cov_matrix=-1, iteration_count=100000):
    # sets forced_look_back to LOOK_BACK if no price list is shorter than LOOK_BACK;
    # if any price list is shorter than LOOK_BACK, sets forced_look_back to the length of the shortest price list
    if cov_matrix == -1:
        calc_cov_matrix = True
    else:
        calc_cov_matrix = False

    forced_look_back = look_back

    for asset_class in asset_classes:
        if len(asset_class['price_list']) < forced_look_back:
            forced_look_back = len(asset_class['price_list'])

    # initialize the covariance matrix, if one is not given in the parameter
    if calc_cov_matrix:
        price_list_array = np.zeros(shape=(len(asset_classes), forced_look_back - 1))
        for i in range(len(asset_classes)):
            price_list_array[i] = price_series_to_return_series(asset_classes[i]['price_list'][-forced_look_back:])
        cov_matrix = np.cov(price_list_array)
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

        portfolio_vol = np.sqrt(252) * np.sqrt(np.dot(rand_weights.T, np.dot(cov_matrix, rand_weights)))

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

    if print_results:
        portfolios_to_print = 5
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


# Takes a list of prices, and returns a list of one less size with each entry representing
# ln(price_series[i+1]/prices_series[i])
def price_series_to_return_series(price_series):
    return_series = []

    for i in range(len(price_series)-1):
        return_series.append(np.log(price_series[i+1]/price_series[i]))

    return return_series


def calc_portfolio_vol_two_assets(weights, standard_deviations, correlation):
    portfolio_variance = (weights[0]**2)*(standard_deviations[0]**2)
    portfolio_variance += (weights[1]**2)*(standard_deviations[1]**2)

    portfolio_variance += weights[0]*weights[1]*standard_deviations[0]*standard_deviations[1]*correlation
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_volatility


# Calculates covariance of the daily percent changes ln(day_2/day_1) and so on, assuming that the
# average is 0, NOT using the manually calculated average daily percent change
# like with variance calcs, we use the SAMPLE covariance formula, i.e. (n-1) in denominator not n
def calc_covariance(prices_one, prices_two, days=-1):
    if days == -1:
        prices_one = prices_one.copy()
        prices_two = prices_two.copy()
    else:
        if days > len(prices_one):
            raise ValueError("The look back is greater than the number of prices in the first list.")
        if days > len(prices_two):
            raise ValueError("The look back is greater than the number of prices in the second list.")
        prices_one = prices_one.copy()[-1*days:]
        prices_two = prices_two.copy()[-1*days:]

    # need to confirm that the first and second list of prices are of equal length
    # whichever list is longer is shortened by ditching the LAST set of entries that are in excess of the other list
    if len(prices_one) > len(prices_two):
        prices_one = prices_one[0:len(prices_two)]
    elif len(prices_two) > len(prices_one):
        prices_two = prices_two[0:len(prices_one)]

    numerator = 0
    for i in range(0, len(prices_one)-1):
        numerator += np.log(prices_one[i+1]/prices_one[i])*np.log(prices_two[i+1]/prices_two[i])

    covariance = numerator/(len(prices_one)-2)
    return covariance


# Takes in 1D numpy array of prices and integer amount of days to calculate realized volatility over
# takes prices, a list of prices, and days as the look back
# note: uses days to take the LAST 'days' entries in prices to calculate annualized RV over
# ASSUMES these prices are daily (else we are improperly annualized)
# https://www.realvol.com/VolFormula.htm
def calc_annualized_realized_vol(prices, days=-1):
    if days == -1:
        days = len(prices)
    if days > len(prices):
        raise ValueError("Look back can't be greater than number of days in the data!")
    prices = prices.copy()[-1*days:]
    Rt_squared_sum = 0
    for i in range(len(prices) - 1):
        Rt_squared_sum += np.log(prices[i+1] / prices[i]) ** 2

    return np.sqrt((TRADING_DAYS_PER_YEAR / (len(prices) - 2)) * Rt_squared_sum)


# Takes list of (daily) prices and an optional day count "days" to limit the calculation to
# Calculates arithmetic mean of daily percent change
# Arithmetic mean of daily return -> avg daily return -> avg daily return * 252 -> annualized_avg_return
# 252 above is the typical trading days per year (and is indeed handled internally as TRADING_DAYS_PER_YEAR)
def calc_annualized_arithmetic_mean(prices, days=-1):
    if days == -1:
        days = len(prices)
    if days > len(prices):
        raise ValueError("Look back can't be greater than number of days in the data!")
    prices = prices.copy()[-1 * days:]

    sum_of_percent_changes = 0
    for i in range(len(prices)-1):
        sum_of_percent_changes += prices[i+1]/prices[i]

    average_daily_return = sum_of_percent_changes/(len(prices)-1)
    annualized_return = average_daily_return ** TRADING_DAYS_PER_YEAR

    return annualized_return


# Takes in a list of prices and a number of days to look back;
# takes the last 'look back' entries of prices and calculates the annualized geometric mean
# ASSUMES these prices are daily (else we improperly annualized)
def calc_annualized_geometric_mean(prices, look_back=-1):
    if look_back == -1:
        look_back = len(prices)
    if look_back > len(prices):
        raise ValueError("Look back can't be greater than number of days in the data!")
    prices = prices.copy()[-1*look_back:]
    geometric_mean = 1
    for i in range(len(prices) - 1):
        geometric_mean *= (prices[i+1]/prices[i])
    geometric_mean = geometric_mean**(1/(len(prices)-1))
    return geometric_mean**TRADING_DAYS_PER_YEAR


# Takes 1D numpy array of prices and integer amount of days to calc realized vol over,
# and then it calculates, based on every period of days over the prices,
# the mean and standard deviation of RV over that period
# Ex: 300 days of prices, 30 day RV, takes first 30 days, then day 1 to day 31, then day 2 to day 32,
# and so on, calculating RV over those 30, then for all 30 day periods returns mean and std of the RVs
def calc_rv_stats(prices, look_back):
    if look_back > len(prices):
        raise ValueError("Look back can't be greater than number of days in the data!")
    look_back_rvs = []
    for i in range(len(prices) - look_back + 1):
        temp_prices = []
        for j in range(i, look_back + i):
            temp_prices.append(prices[j])
        look_back_rvs.append(calc_annualized_realized_vol(temp_prices, look_back))
    look_back_rvs = np.array(look_back_rvs)
    return np.mean(look_back_rvs), np.std(look_back_rvs)


# TODO: This calculates each period's mean return over any given 30-day period as day 30 prices / day 1 price
# TODO: is that a correct interpretation of "30 day mean return" given we are using 30 day volatility figures?
# Takes 1D numpy array of prices and integer amount of days to calc mean over,
# and then it calculates, based on every period of days over the prices,
# the mean and standard deviation of the mean over that period
# Ex: 300 days of prices, 30 day mean, takes first 30 days, then day 1 to day 31, then day 2 to day 32,
# and so on, calculating MEAN over those 30, then for all 30 day periods returns mean and std of the MEANs
def calc_mean_stats(prices, look_back):
    if look_back > len(prices):
        raise ValueError("Look back can't be greater than number of days in the data!")
    look_back_means = []
    for i in range(len(prices) - look_back + 1):
        temp_prices = []
        for j in range(i, look_back + i):
            temp_prices.append(prices[j])
        look_back_means.append(temp_prices[-1]/temp_prices[0])
    look_back_means = np.array(look_back_means)
    return np.mean(look_back_means), np.std(look_back_means)


# TODO: Find a better name for this function (or not make it a function since it's one line anyway)
# Given a 1D numpy array of prices, an integer number of days to look back, and a particular value,
# returns what portion of the regime it's in, **ASSUMING STANDARD NORMAL**
# For example, if 30 day RV is generally 0.30, with a low STD and the value input is 0.10, one should expect
# that this 30 day RV is in the bottom 20% (return 0.2), or something
def calc_standard_normal_cum_prob(prices, look_back, value):
    stats = calc_rv_stats(prices, look_back)
    mean = stats[0]
    std = stats[1]
    return norm.cdf((value-mean)/std)


# Takes two lists of prices and returns their covariance
def calc_covariance_and_correlation(prices_one, prices_two, look_back=-1):
    if look_back == -1:
        prices_one = prices_one.copy()
        prices_two = prices_two.copy()
    else:
        if look_back > len(prices_one):
            raise ValueError("The look back is greater than the number of prices in the first list.")
        if look_back > len(prices_two):
            raise ValueError("The look back is greater than the number of prices in the second list.")
        prices_one = prices_one.copy()[-1*look_back:]
        prices_two = prices_two.copy()[-1*look_back:]

    # need to confirm that the first and second list of prices are of equal length
    # whichever list is longer is shortened by ditching the LAST set of entries that are in excess of the other list
    if len(prices_one) > len(prices_two):
        prices_one = prices_one[0:len(prices_two)]
    elif len(prices_two) > len(prices_one):
        prices_two = prices_two[0:len(prices_one)]

    mean_one = np.mean(prices_one)
    mean_two = np.mean(prices_two)
    covariance = 0
    for i in range(len(prices_one)):
        covariance += (prices_one[i] - mean_one)*(prices_two[i] - mean_two)

    covariance *= (1/(len(prices_one)-1))

    correlation = covariance / (np.std(prices_one)*np.std(prices_two))

    return covariance, correlation


# Takes in a csv file name as a sting (Ex: csv = 'SPY.csv') and prints a list of the column titles
def print_col_titles(csv):
    data = pd.read_csv(csv)
    print(list(data.columns))
