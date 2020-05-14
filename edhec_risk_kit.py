import pandas as pd
import numpy as np
import scipy.stats
import ipywidgets as widgets
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


def get_ind_size():
    '''
    Load and format the Ken French 30 Industry Portfolio Sizes
    '''
    ind = pd.read_csv('ind30_m_size.csv',
                  header=0,
                 index_col=0,
                 parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_nfirms():
    '''
    Load and format the Ken French 30 Industry Portfolio number of firms
    '''
    ind = pd.read_csv('ind30_m_nfirms.csv',
                  header=0,
                 index_col=0,
                 parse_dates=True)
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def get_total_market_index_returns():
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis=0)
    total_market_return = (ind_capweight * ind_return).sum(axis=1)
    return total_market_return


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
 

def gmv(cov):
    '''
    Returns the weight of the global minimum vol portfolio
    given covariance matrix
    '''
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)
    
    
def plot_ef(n_points, er, cov, riskfree_rate=0, show_cml=False, style='.-', show_ew=False, show_gmv=False):
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
    ax = ef.plot.line(x='Volatility', y='Returns', style=style)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # display ew
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # display ew
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
    if show_cml:
        ax.set_xlim(left = 0)
        # get msr weights first
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add capital market line with msr point
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', markersize=12, linewidth=2)
    return ax


def msr(riskfree_rate, er, cov):
    '''
    Returns weights of the portfolio that gives maximum sharpe ratio given
    riskfree rate, expected returns, covariance matrix
    '''
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights)-1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        '''
        returns the negative of the sharpe ratio, given weights
        '''
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    results = minimize(neg_sharpe_ratio,
                       init_guess,
                       args=(riskfree_rate, er, cov,),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                      )
    return results.x


def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03, drawdown=None):
    '''
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    returns a dictionary containing: asset value history, risk budget history, risky weight history
    '''
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start
    
    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=['R'])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        # assign all values as a fraction of riskfree rate
        safe_r.values[:] = riskfree_rate/12
    # intermediate dataframes of same size to fill intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        # recompute account value at the end of this step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save histories for analytics and plotting
        account_history.iloc[step] = account_value
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        'Wealth': account_history,
        'Risky Wealth': risky_wealth,
        'Risk Budget': cushion_history,
        'Risky Allocation': risky_w_history,
        'm': m,
        'start': start,
        'floor': floor,
        'risky_r': risky_r,
        'safe_r': safe_r
    }
    return backtest_result


def summary_stats(r, riskfree_rate=0.03):
    '''
    Return a dataframe that contains aggregated summary stats for the returns in the column of r
    '''
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = r.aggregate(sharpe_ratio, riskfree_rate=riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        'Annualized Return': ann_r,
        'Annualized Vol': ann_vol,
        'Skewness': skew,
        'Kurtosis': kurt,
        'Cornish-Fisher VaR (5%)': cf_var5,
        'Historic CVaR (5%)': hist_cvar5,
        'Sharpe Ratio': ann_sr,
        'Max Drawdown': dd
    })


def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    '''
    Evolution of a stock price using a Geometric Brownian Motion model.
    '''
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year)
    # standard way
    #rets_plus_1 = np.random.normal(loc=(1+mu*dt),scale=(sigma*np.sqrt(dt)), size=(n_steps, n_scenarios))
    # without discretization error
    rets_plus_1 = np.random.normal(loc=(1+mu)**dt,
                                   scale=(sigma*np.sqrt(dt)),
                                   size=(n_steps, n_scenarios)
                                  )
    rets_plus_1[0] = 1
    # to prices
    ret_val = s_0*pd.DataFrame(rets_plus_1).cumprod() if prices else rets_plus_1-1
    return ret_val


def show_gbm(n_scenarios, mu, sigma):
    '''
    Draw the results of a stock price evolution under a Geometric Brownian Motion model
    '''
    s_0 = 100
    prices = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, s_0=s_0)
    ax = prices.plot(legend=False, color='indianred', alpha=0.5, linewidth=2, figsize=(12,5))
    ax.axhline(y=s_0, ls=':', color='black')
    ax.set_ylim(top=400)
    # draw a dot at the origin
    ax.plot(0, s_0, marker='o', color='darkred', alpha=0.2)
    

def show_cppi(n_scenarios=50, mu=0.07, sigma=0.15, m=3, floor=0., riskfree_rate=0.03, y_max=100):
    '''
    Plot the results of a Monte Carlo Simulation of CPPI
    '''
    start = 100
    sim_rets = gbm(n_scenarios=n_scenarios, mu=mu, sigma=sigma, prices=False, steps_per_year=12)
    risky_r = pd.DataFrame(sim_rets)
    # run the "backtest"
    btr = run_cppi(risky_r=risky_r, riskfree_rate=riskfree_rate, m=m, start=start, floor=floor)
    wealth = btr['Wealth']
    
    # calculate terminal wealth stats
    y_max = wealth.values.max()*y_max/100
    terminal_wealth = wealth.iloc[-1]
    
    tw_mean = terminal_wealth.mean()
    tw_median = terminal_wealth.median()
    failure_mask = np.less(terminal_wealth, start*floor)
    n_failures = failure_mask.sum()
    p_fail = n_failures/n_scenarios
    
    e_shortfall = np.dot(terminal_wealth-start*floor, failure_mask)/n_failures if n_failures > 0 else 0.0
    
    # plot
    fig, (wealth_ax, hist_ax) = plt.subplots(nrows=1, ncols=2, sharey=True, gridspec_kw={'width_ratios':[3, 2]}, figsize=(24, 9))
    plt.subplots_adjust(wspace=0.0)
    
    wealth.plot(ax=wealth_ax, legend=False, alpha=0.3, color='indianred')
    wealth_ax.axhline(y=start, ls=':', color='black')
    wealth_ax.axhline(y=start*floor, ls='--', color='red')
    wealth_ax.set_ylim(top=y_max)
    
    terminal_wealth.plot.hist(ax=hist_ax, bins=50, ec='w', fc='indianred', orientation='horizontal')
    hist_ax.axhline(y=start, ls=':', color='black')
    hist_ax.axhline(y=tw_mean, ls=':', color='blue')
    hist_ax.axhline(y=tw_median, ls=':', color='purple')
    hist_ax.annotate(f'Mean: ${int(tw_mean)}', xy=(.7, .9), xycoords='axes fraction', fontsize=24)
    hist_ax.annotate(f'Median: ${int(tw_median)}', xy=(.7, .85), xycoords='axes fraction', fontsize=24)
    if floor > 0.01:
        hist_ax.axhline(y=start*floor, ls='--', color='red', linewidth=3)
        hist_ax.annotate(f'Violations: {n_failures} ({p_fail*100:2.2f}%)\nE(shortfall)=${e_shortfall:2.2f}', xy=(.7, .7), xycoords='axes fraction', fontsize=24)
    
cppi_controls = widgets.interactive(show_cppi,
                                   n_scenarios=widgets.IntSlider(min=1, max=1000, step=5, value=50),
                                   mu=(0., +.2, .01),
                                   sigma=(0, .3, .05),
                                   floor=(0, 2, .1),
                                   m=(1, 5, .5),
                                   riskfree_rate=(0, .05, .01),
                                   y_max=widgets.IntSlider(min=0, max=100, step=1, value=100, description='Zoom Y Axis')
                                   )


def discount(t, r):
    '''
    Compute the price of a pure discount bond that pays a dollar at time t, given interest rate t
    '''
    return (1+r)**(-t)


def pv(l, r):
    '''
    Compute present value of a sequence of liabilities
    l is indexed by the time, and values are the amounts of each liability
    returns present value
    '''
    dates = l.index
    discounts = discount(dates, r)
    return (discounts*l).sum()


def funding_ratio(assets, liabilities, r):
    '''
    Computes funding ratio of some assets given liabilities and interest rate
    '''
    return assets/pv(liabilities, r)