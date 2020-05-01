import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm


def drawdown(return_series: pd.Series):
    '''
    Takes a time series of asset returns,
    computes and returns a dataframe that contains:
    the wealth index
    the previous peaks
    percent drawdowns
    '''
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index-previous_peaks)/previous_peaks
    return pd.DataFrame({
        'wealth': wealth_index,
        'peaks': previous_peaks,
        'drawdown': drawdowns
    })


def get_ffme_returns():
    '''
    Load the Fama-French dataset for returns of the top
    and bottom deciles by market cap
    '''
    me_m = pd.read_csv('Portfolios_Formed_on_ME_monthly_EW.csv',
                  header=0,
                  index_col=0,
                  parse_dates=True,
                  na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format='%Y%m').to_period('M')
    return rets


def  get_hfi_returns():
    '''
    Load and format the EDHEC Hedge Fund Index returns
    '''
    hfi = pd.read_csv('edhec-hedgefundindices.csv',
                     header=0,
                     index_col=0,
                     parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi


def semideviation(r):
    '''
    Returns semideviation, aka negative semideviation of r
    r must be series or dataframe
    '''
    is_negative = r<0
    return r[is_negative].std(ddof=0)


def skewness(r):
    '''
    scipy.stats.skew() implementation exists, an alternate:
    computes skewness of series or dataframe
    returns series or float
    '''
    demeaned_r = r - r.mean()
    # use population standard deviation as opposed to sample standard deviation, degrees of freedom=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    '''
    scipy.stats.kurtosis() implementation exists, an alternate:
    computes skewness of series or dataframe
    returns series or float
    '''
    demeaned_r = r - r.mean()
    # use population standard deviation as opposed to sample standard deviation, degrees of freedom=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def is_normal(r, level=0.01):
    '''
    Applies th Jarque Bera test to determine if Series is normal or not
    test is applied at 1% level by default
    returns true if hypothesis of normality is accepted, false otherwise
    '''
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level


def var_historic(r, level=5):
    '''
    Returns historic Value at Risk at a specified level
    returns the number such that level percent of the returns fall below
    that number
    '''
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r to be Series or DataFrame.')
        

def var_gaussian(r, level=5, modified=False):
    '''
    returns parametric Gaussian VaR of a Series or DataFrame.
    If modified=True, returns modified VaR using
    Cornish-Fisher modification
    '''
    # compute Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
             (z**2 - 1)*s/6 +
             (z**3 - 3*z)*(k-3)/24 -
             (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))


def cvar_historic(r, level=5):
    '''
    computes conditional VaR of Series or DataFrame
    '''
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError('Expected r to be a Series or DataFrame.')