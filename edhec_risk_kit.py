import pandas as pd
import numpy as np
import scipy.stats
from scipy.stats import norm
from scipy.optimize import minimize


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


def get_ind_returns():
    '''
    Load and format the Ken French 30 Industry Portfolio Value Weighted Monthly Returns
    '''
    ind = pd.read_csv('ind30_m_vw_rets.csv',
                  header=0,
                 index_col=0,
                 parse_dates=True)/100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def annualize_rets(r, periods_per_year):
    '''
    Annualizes the returns
    '''
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    '''
    Annualizes the volatility of a set of returns
    '''
    return r.std()*(periods_per_year**0.5)


def sharpe_ratio(r, riskfree_rate, periods_per_year):
    '''
    Computes the annualized sharpe ratio of a set of returns
    '''
    # convert annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


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
        

def portfolio_return(weights, returns):
    '''
    weights -> returns
    '''
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    '''
    weights -> vol
    '''
    return (weights.T @ covmat @ weights)**0.5


def plot_ef2(n_points, er, cov, style='.-'):
    '''
    Plots the 2-asset efficient frontier
    '''
    if er.shape[0] != 2:
        raise ValueError('plot_ef2 can only plot 2-asset frontiers.')
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    return ef.plot.line(x='Volatility', y='Returns', style=style)


def minimize_vol(target_return, er, cov):
    '''
    target_return -> w
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    results = minimize(portfolio_vol,
                       init_guess,
                       args=(cov,),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds
                      )
    return results.x


def optimal_weights(n_points, er, cov):
    '''
    -> list of weights to run the optimizer on to minimize the vol on
    '''
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights
 

def plot_ef(n_points, er, cov, style='.-'):
    '''
    Plots the N-asset efficient frontier
    '''
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    return ef.plot.line(x='Volatility', y='Returns', style=style)