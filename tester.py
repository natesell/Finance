import math
import funcs
import pandas as pd
import numpy as np
import sys

csv_data = pd.read_csv('Excel Sheets/SPY_current.csv')
spy_adj_close = csv_data["Adj Close"].to_numpy()[:-1]

csv_data = pd.read_csv('Excel Sheets/TLT_current.csv')
tlt_adj_close = csv_data["Adj Close"].to_numpy()[:-1]

'''
# Initializing list of dictionaries
lis = [{"name": "Nandini", "age": 20},
        {"name": "Manjeet", "age": 20},
        {"name": "Nikhil", "age": 19}]

# using sorted and lambda to print list sorted
# by age
print(lis)
"The list printed sorting by age: "
print(sorted(lis, key=lambda i: i['age']))
'''
# https://www.multpl.com/s-p-500-earnings-yield
# https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield

LOOK_BACK = 90
SPY_return = 0.027
TLT_return = 0.01475

csv_data = pd.read_csv('Excel Sheets/SPY_current.csv')
spy_adj_close = csv_data["Adj Close"].to_numpy()

csv_data = pd.read_csv('Excel Sheets/TLT_current.csv')
tlt_adj_close = csv_data["Adj Close"].to_numpy()

asset_classes = [
    {'ticker': "SPY", 'annual_return': SPY_return, 'price_list': spy_adj_close},
    {'ticker': "TLT", 'annual_return': TLT_return, 'price_list': tlt_adj_close}
]

funcs.portfolio_optimizer(asset_classes, LOOK_BACK, iteration_count=500000, graph_results=True)
