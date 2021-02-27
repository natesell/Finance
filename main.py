import funcs
from funcs import YahooFinanceHistory
import numpy as np
import matplotlib as plt
import csv
import pandas as pd
from datapackage import Package
import json

# NOTE 'datapackage' is something I am unfamiliar with but the following is based on
# the wonderful work from https://datahub.io/core/s-and-p-500-companies-financials#python

'''
TRADING_DAYS_PER_YEAR = funcs.TRADING_DAYS_PER_YEAR
SPY_CSV = 'SPY.csv'
VIX_CSV = '^VIX.csv'

csv_data = pd.read_csv(SPY_CSV)
vix_data = pd.read_csv(VIX_CSV)
spy_adj_close = csv_data["Adj Close"].to_numpy()
vix_adj_close = vix_data["Adj Close"].to_numpy()

print(f"Realized vol for SPY over the last 30 days is {funcs.calc_realized_vol(spy_adj_close, 30)}")
print(f"Realized vol for VIX over the last 30 days is {funcs.calc_realized_vol(vix_adj_close, 30)}")
print(f"Based on {len(vix_adj_close)} days of VIX adjusted close data, the mean and std of 30 day RV is: ")

vix_look_back_rv_stats = funcs.calc_rv_stats(vix_adj_close, 30)
print(f"Mean: {vix_look_back_rv_stats[0]}; STD: {vix_look_back_rv_stats[1]}")

vix_percent_of_regime = funcs.calc_standard_normal_cum_prob(vix_adj_close, 30,
                                                            funcs.calc_realized_vol(vix_adj_close, 30))
print(f"From the above, we can conclude that 30 day RV is currently in the bottom {vix_percent_of_regime*100}%.")
'''

package = Package('https://datahub.io/core/s-and-p-500-companies-financials/datapackage.json')

# print processed tabular data (if exists any)
for resource in package.resources:
    if resource.descriptor['datahub']['type'] == 'derived/csv':
        snp_companies = resource.read()
        break

num_of_snp_companies = len(snp_companies)
print(num_of_snp_companies)
# Really bad title; creating a list of dicts where each dict represents a company
# Ex: {"symbol" = "MMM", "name" = 3M Company, "change" = 2.5} where change is
# last year % change in price
companies_performance = []
snp_companies_copy = snp_companies.copy()
look_back = 365
test_performance_list_of_strings = []
iterations = 0
while (len(companies_performance) < num_of_snp_companies) and iterations < 10:
    count = 1
    for company in snp_companies_copy:
        symbol = company[0]

        try:
            df = YahooFinanceHistory(symbol, days_back=look_back).get_quote()
            adj_close = list(df["Adj Close"])
            change = (adj_close[-1] - adj_close[0]) / adj_close[0]
            companies_performance.append({"symbol": symbol, "name": company[1], "change": change})
            snp_companies_copy.remove(company)
            print(f"{count}.) {symbol}")
        except:
            print(f'{count}.) Failed for {symbol}')

        count += 1

    iterations += 1
    test_performance_list_of_strings.append(f"After {iterations} iterations, there were "
                                            f"{len(companies_performance)} companies.")

for string in test_performance_list_of_strings:
    print(string)
print(f"Hence we are still missing {num_of_snp_companies-len(companies_performance)} companies!")

for company in snp_companies_copy:
    print(company)

with open('companies_and_performance.json', 'w') as f:  # writing JSON object
    json.dump(companies_performance, f)

# If you need to know how to read from json, here is an example
'''
try:
    with open('movie_database.json', 'r') as f:
        database = json.load(f)   
except:
    database = []
'''