import funcs


# GOAL: Monte-Carlo-esque simulation to approximate best leverage combinations
#       placed on two asset classes with differing mean return, variance, and covariance
#       in some crude attempt to understand proper leverage placement on a stock / bond difference

# Glossary-type thing, control + F something to learn more about it
# --------------------------------------------
# classes:
# 'AssetClass' represents something with a mean and variance

# methods:
# 'calc_covariance_and_correlation' calculates the covariance and calculation between two AssetClass objects
# ---------------------------------------------

"""
description:
AssetClass object emulates a stock or bond
mean and variance are annualized, and the mean is a geometric mean 
make certain if setting mean and variance manually that their look back is equivalent 
    (unless your use case is weird)
**HENCE: mean and variance refer to the last look_back days, not a long term average of all look_back length
         time periods

declaration:
mean = annualized geometric mean over look_back days
variance = annualized variance calculated over look_back days 
    optional parameters:
    look_back = number of days of a look back that mean and variance are calculated relative to
        automatically -1 if no declaration
    prices = list of prices
        automatically -1 if no declaration
    calc_stats = boolean which tells AssetClass to calculate mean and variance or use given parameters
        automatically False if no declaration

methods:

properties:
(see declaration for additional properties)

"""


class AssetClass:
    def __init__(self, mean, variance, look_back=-1, prices=-1, calc_stats=False):
        self.mean = mean
        self.variance = variance
        self.look_back = look_back
        self.prices = prices
        self.calc_stats = calc_stats
        if calc_stats:
            self.mean = funcs.calc_annualized_geometric_mean(prices, look_back)
            self.variance = funcs.calc_annualized_realized_vol(prices, look_back)**2


# Takes two AssetClass objects and calculates the covariance and correlation between their values
# first_asset and second_asset should be two distinct AssetClass objects;
# look_back can be specified if you'd like to calculate the covariance/correlation during a certain look back
# from the last similar number of prices from first_asset and second_asset
# NOTE: This means by default, the method will use the maximum number of entries that both AssetClass objects share
def calc_covariance_and_correlation(first_asset, second_asset, look_back=-1):
    if first_asset.prices == -1:
        raise ValueError("The first asset has a blank list of prices.")
    if second_asset.prices == -1:
        raise ValueError("The second asset has a blank list of prices.")

    return funcs.calc_covariance_and_correlation(first_asset.prices, second_asset.prices, look_back)


a = AssetClass(2, 5)
b = AssetClass(3, 6)

