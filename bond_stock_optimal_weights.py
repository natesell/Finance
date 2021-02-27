import math
import funcs
import pandas as pd
import numpy as np
import sys

# 11/9/2020

# TODO: (1) Plot sharpe, bond prop, stock prop to see clusters (scatter plot?)

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

# SPY I calculate earnings yield, BONDs just straight rip 30 year treasury bill yield
SPY_return = 0.0279  # https://www.multpl.com/s-p-500-earnings-yield
BOND_return = 0.0173  # https://www.treasury.gov/resource-center/data-chart-center/interest-rates/pages/textview.aspx?data=yield
# https://www.ishares.com/us/products/239454/ishares-20-year-treasury-bond-etf -> Average Yield to Maturity

# make a function in funcs for arbitrarily large amount of variables?
# prolly too hard kek
SPY_and_TLT_covariance = funcs.calc_covariance(spy_adj_close, tlt_adj_close, LOOK_BACK)
SPY_and_TLT_correlation = SPY_and_TLT_covariance / (spy_rv * tlt_rv)

print("90-day look back SPY rv: " + str(spy_rv) + " | Ripped SPY Annualized Return: " +
      str(SPY_return))
print("90-day look ack BOND rv: " + str(tlt_rv) + " | Ripped BOND Annualized Return: " +
      str(BOND_return))
print("Based on a 90-day look back, correlation between stocks and bonds is: " + str(SPY_and_TLT_correlation))

largest_sharpe_ratio = -sys.maxsize  # want a possible negative sharpe ratio to still be a possible "largest" candidate
sharpe_ratios = []  # dicts should have format {"sharpe_ratio": x, "stock_proportion": y, "bond_proportion": z}
for stock_proportion in range(0, 101):
    stock_proportion *= (1 / 100)
    stock_proportion = round(stock_proportion, 2)
    bond_proportion = round(1 - stock_proportion, 2)

    portfolio_return = ((stock_proportion * SPY_return) + (bond_proportion * BOND_return))
    # http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-portfolio-I.pdf
    portfolio_vol = funcs.calc_portfolio_vol_two_assets([stock_proportion, bond_proportion], [spy_rv, tlt_rv],
                                                              SPY_and_TLT_correlation)

    # can use BOND_return comment link -> 10 year bonds as risk free rate if doing sharpe (mean - risk_free) / vol
    sharpe_ratio = portfolio_return / portfolio_vol
    sharpe_ratios.append({"sharpe_ratio": sharpe_ratio, "stock_proportion": stock_proportion,
                          "bond_proportion": bond_proportion, 'portfolio_return': portfolio_return,
                          'portfolio_vol': portfolio_vol})

sharpe_ratios = sorted(sharpe_ratios, key=lambda i: i['sharpe_ratio'])
print(sharpe_ratios[-5:])
