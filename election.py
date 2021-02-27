import math
import funcs
import pandas as pd
import numpy as np

# check oct 15 to jan 15 for past few elections
# calc annualized rv of those periods
# average it
# see what jan 15 spy options IV are
# if avg rv of those is significantly less than what IV is
# sell vol, else buy vol / re-evaluate

time_periods = ['2000-2001', '2004-2005', '2008-2009', '2012-2013', '2016-2017']

print("S&P:")
print("--------------------------")
for time_period in time_periods:
    SnP_CSV = 'Excel Sheets/S&P Elections/' + time_period + '.csv'
    csv_data = pd.read_csv(SnP_CSV)
    spy_adj_close = csv_data["Adj Close"].to_numpy()
    rv = funcs.calc_annualized_realized_vol(spy_adj_close)

    print("The realized vol from Oct 15 to Jan 15 during " + time_period
          + " is " + str(rv))


rvs = []

print("\nSPY:")
print("--------------------------")
for time_period in time_periods:
    SPY_CSV = 'Excel Sheets/SPY Elections/' + time_period + '.csv'
    csv_data = pd.read_csv(SPY_CSV)
    spy_adj_close = csv_data["Adj Close"].to_numpy()
    rv = funcs.calc_annualized_realized_vol(spy_adj_close)
    rvs.append(rv)

    print("The realized vol from Oct 15 to Jan 15 during " + time_period
          + " is " + str(rv))

print("The average realized vol from Oct 15 to Jan 15 during elections is " + str(np.mean(rvs)))
rvs_without_2008 = rvs.copy()
del rvs_without_2008[2]
print("Without 2008, the average realized falls to " + str(np.mean(rvs_without_2008)))
