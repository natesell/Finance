import funcs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This takes all TLT data (and SPY data just as far back), then converts it to percent change
# then iterating from a 2 day look-back to a 300 day look-back; for each look-back
# tests ITERATIONS_PER_LOOK_BACK_DAY different starting days, comparing how
# useful past look-back is on predicting forward 30-day information (stock vol, bond vol, correlation)

csv_data = pd.read_csv('Excel Sheets/Backtesting/SPY_long.csv')
spy_adj_close = funcs.price_series_to_return_series(csv_data["Adj Close"].to_numpy())

csv_data = pd.read_csv('Excel Sheets/Backtesting/TLT_long.csv')
tlt_adj_close = funcs.price_series_to_return_series(csv_data["Adj Close"].to_numpy())

LOOK_BACK_DAYS_TO_TEST = 300
ITERATIONS_PER_LOOK_BACK_DAY = 1000
FORWARD_LOOK = 30

results = []
for look_back in range(2, LOOK_BACK_DAYS_TO_TEST + 1):
    predicted_stock_vol = []
    forward_thirty_day_stock_vol = []

    predicted_bond_vol = []
    forward_thirty_day_bond_vol = []

    predicted_covariance = []
    forward_thirty_day_covariance = []
    for i in range(ITERATIONS_PER_LOOK_BACK_DAY):
        starting_day = np.random.randint(len(spy_adj_close) - (100 + LOOK_BACK_DAYS_TO_TEST)) + \
                       (LOOK_BACK_DAYS_TO_TEST + FORWARD_LOOK + 20)  # ensures 30 days of data forward, >300 backward
        backward_cov_matrix = np.cov(np.array([spy_adj_close[starting_day - look_back:starting_day],
                                               tlt_adj_close[starting_day - look_back:starting_day]]))
        forward_cov_matrix = np.cov(np.array([spy_adj_close[starting_day:starting_day+FORWARD_LOOK],
                                              tlt_adj_close[starting_day:starting_day+FORWARD_LOOK]]))

        predicted_stock_vol.append(np.sqrt(backward_cov_matrix[0][0]))
        forward_thirty_day_stock_vol.append(np.sqrt(forward_cov_matrix[0][0]))

        predicted_bond_vol.append(np.sqrt(backward_cov_matrix[1][1]))
        forward_thirty_day_bond_vol.append(np.sqrt(forward_cov_matrix[1][1]))

        predicted_covariance.append(backward_cov_matrix[0][1])
        forward_thirty_day_covariance.append(forward_cov_matrix[0][1])

    results.append({"look_back": look_back,
                    "stock_vol_corr": np.corrcoef(np.array([predicted_stock_vol, forward_thirty_day_stock_vol]))[0][1],
                    "bond_vol_corr": np.corrcoef(np.array([predicted_bond_vol, forward_thirty_day_bond_vol]))[0][1],
                    "covariance_corr": np.corrcoef(np.array([predicted_covariance, forward_thirty_day_covariance]))[0][1]
                    })

look_backs = []
stock_vol_corrs = []
bond_vol_corrs = []
covariance_corrs = []
for result in results:
    look_backs.append(result["look_back"])
    stock_vol_corrs.append(result["stock_vol_corr"])
    bond_vol_corrs.append(result["bond_vol_corr"])
    covariance_corrs.append(result["covariance_corr"])


plt.xlabel("Look Back")
plt.ylabel("Predictability")
plt.plot(look_backs, stock_vol_corrs, color='red', label='stock volatility')
plt.plot(look_backs, bond_vol_corrs, color='blue', label='bond volatility')
plt.plot(look_backs, covariance_corrs, color='green', label='stock & bond covariance')
plt.legend()
plt.show()
print('a')