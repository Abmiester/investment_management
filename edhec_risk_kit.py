import pandas as pd
import numpy as np
import scipy.stats
import ipywidgets as widgets
import statsmodels.api as sm
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


def get_fff_returns():
    '''
    Load the Fama-French research data monthly factors
    '''
    ffm = pd.read_csv('F-F_Research_Data_Factors_m.csv',
                     header=0,
                     index_col=0,
                     parse_dates=True)
    ffm = ffm/100
    ffm.index = pd.to_datetime(ffm.index, format='%Y%m').to_period('M')
    return ffm


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


def get_ind_file(filetype, weighting='vw', n_inds=30):
    '''
    Load and format the Ken French Industry Portfolio files
    Variant is a tuple of (weighting, size) where:
        weighting is one of 'ew', 'vw'
        number of inds is 30 or 49
    '''
    if filetype is 'returns':
        name = f'{weighting}_rets'
        divisor = 100
    elif filetype is 'nfirms':
        name = 'nfirms'
        divisor = 1
    elif filetype is 'size':
        name = 'size'
        divisor = 1
    else:
        raise ValueError(f'filetype must be one of: returns, nfirms, size.')
    
    ind = pd.read_csv(f'ind{n_inds}_m_{name}.csv',
                     header=0,
                     index_col=0,
                     na_values=-99.99)/divisor
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind


def get_ind_returns(weighting='vw', n_inds=30):
    '''
    Load and format the Ken French Industry Portfolios Monthly Returns
    '''
    return get_ind_file('returns', weighting=weighting, n_inds=n_inds)


def get_ind_nfirms(n_inds=30):
    '''
    Load and format the Ken French Industry Portfolios Average Number of Firms
    '''
    return get_ind_file('nfirms', n_inds=n_inds)


def get_ind_size(n_inds=30):
    '''
    Load and format the Ken French Industry Portfolios Average Size (Market Cap)
    '''
    return get_ind_file('size', n_inds=n_inds)


def get_ind_market_caps(n_inds=30, weights=False):
    '''
    Load the industry portfolio data and derive the market caps
    '''
    ind_nfirms = get_ind_nfirms(n_inds=n_inds)
    ind_size = get_ind_size(n_inds=n_inds)
    ind_mktcap = ind_nfirms * ind_size
    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis=0)
        return ind_capweight
    return ind_mktcap


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


def compound(r):
    '''
    returns the result of compounding the set of returns in r
    '''
    return np.expm1(np.log1p(r).sum())


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
    list of weights to run the optimizer on to minimize the vol on
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
    Run a backtest of the Constant Proportion Portfolio Insurance (CPPI) strategy,
    given a set of returns for the risky asset
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
    n_steps = int(n_years*steps_per_year) + 1
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
    Computes the price of a pure discount bond that pays a dollar at time t
    and r is the per period interest rate
    returns a |t| * |r| Series or DataFrame
    r can be a float, Series or DataFrame
    returns a DataFrame indexed by t
    '''
    discounts = pd.DataFrame([(r+1)**-i for i in t])
    discounts.index = t
    return discounts


def pv(flows, r):
    '''
    Computes present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    returns present value
    '''
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis=0).sum()


def funding_ratio(assets, liabilities, r):
    '''
    Computes funding ratio of some assets given liabilities and interest rate
    '''
    return pv(assets, r)/pv(liabilities, r)


def inst_to_ann(r):
    '''
    converts short rate to annualized rate
    '''
    return np.expm1(r)


def ann_to_inst(r):
    '''
    convert annualized to short rate
    '''
    return np.log1p(r)


def cir(n_years=10, n_scenarios=1, a=0.05, b=0.03, sigma=0.05, steps_per_year=12, r_0=None):
    '''
    implements the Cox Ingersoll Ross model
    '''
    if r_0 is None:
        r_0 = b
    r_0 = ann_to_inst(r_0)
    dt = 1/steps_per_year
    
    num_steps = int(n_years*steps_per_year)+1
    shock = np.random.normal(0, scale=np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    # for price generation
    h = np.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)
    
    def price(ttm, r):
        _A = ((2*h*np.exp((h+a)*ttm/2))/(2*h+(h+a)*(np.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(np.exp(h*ttm)-1))/(2*h + (h+a)*(np.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P
    prices[0] = price(n_years, r_0)
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        # generate prices at step as well
        prices[step] = price(n_years-step*dt, rates[step])
    
    rates = pd.DataFrame(data=inst_to_ann(rates), index=range(num_steps))
    prices = pd.DataFrame(data=prices, index=range(num_steps))
        
    return rates, prices


def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    '''
    Returns a series of cash flows generated by a bond,
    indexed by a coupon number
    '''
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1, n_coupons+1)
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal
    return cash_flows


def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    '''
    Computes the price of a bond that pays regular coupons until maturity
    at which time the pricipal and the final coupon is returned
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time
    (The index of the discount_rate DataFrame is assumed to be the coupon number)
    '''
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    else:
        # return single time period
        if maturity <= 0:
            return principal + principal * coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate/coupons_per_year)


def macaulay_duration(flows, discount_rate):
    '''
    Computes the Macaulay Duration of a sequence of cash flows
    '''
    discounted_flows = discount(flows.index, discount_rate)*flows
    weights = discounted_flows/discounted_flows.sum()
    return np.average(flows.index, weights=weights)


def match_durations(cf_t, cf_s, cf_l, discount_rate):
    '''
    Returns the weight W in cf_s (short bond) that, along with (1-W) in cf_l (long bond) will have
    an effective duration that matches cf_t (target bond)
    '''
    d_t = macaulay_duration(cf_t, discount_rate)
    d_s = macaulay_duration(cf_s, discount_rate)
    d_l = macaulay_duration(cf_l, discount_rate)
    return (d_l - d_t)/(d_l - d_s)


def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    '''
    Computes the total return on a bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of period
    (e.g. end of 3 months for quarterly dividend)
    and that dividends are reinvested in the bond
    '''
    coupons = pd.DataFrame(data=0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()


def bt_mix(r1, r2, allocator, **kwargs):
    '''
    Runs a back test (simulation) of allocating between two sets of returns,
    r1 and r2 are T x N DataFrames or returns where T is the time step index and N is the number of scenarios.
    allocator is a fuction that takes two sets of returns and allocator specific parameters, and produces
    an allocation to the first portfolio (the rest of the money is invested in GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resultinf N portfolio scenarios
    '''
    if not r1.shape == r2.shape:
        raise ValueError('r1 and r2 need to be the same shape.')
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError('Allocator returned weights that dont match r1')
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix


def fixedmix_allocator(r1, r2, w1, **kwargs):
    '''
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarions
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that:
        each column is a scenario
        each row is the price for a timestamp
    Returns a T x N DataFrame of PSP weights
    '''
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)


def terminal_values(rets):
    '''
    Returns the final values of a dollar at the end of the return period for each scenario
    '''
    return (rets+1).prod()


def terminal_stats(rets, floor=0.8, cap=np.inf, name='Stats'):
    '''
    Produce summary statistics on the terminal values per invested dollar across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time step (we assume rets is sorted by time)
    returns a 1 column DataFrame of summary stats indexed by the stat name
    '''
    terminal_wealth = (rets+1).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= cap
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor-terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (cap-terminal_wealth[reach]).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        'mean': terminal_wealth.mean(),
        'std': terminal_wealth.std(),
        'p_breach': p_breach,
        'e_short': e_short,
        'p_reach': p_reach,
        'e_surplus': e_surplus
    }, orient='index', columns=[name])
    return sum_stats
    
    
def glidepath_allocator(r1, r2, start_glide=1, end_glide=0):
    '''
    Simulates a Target-Date-Fund style gradual move from r1 to r2
    '''
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths


def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    '''
    Allocate between PSP and GHP with the goal to provide exposure to the upside
    of the PSP without violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP.
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    '''
    if zc_prices.shape != psp_r.shape:
        raise ValueError('PSP and ZC Prices must have the same shape.')
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        # PV of floor assuming today's rates and flat YC
        floor_value = floor*zc_prices.iloc[step]
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history


def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    '''
    Allocate between PSP and GHP with the goal of providing exposure to the upside
    of the PSP without violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple
    of the cushion in the PSP.
    Returns a DataFrame with the same shape as the psp/ghp representing the weights in the PSP
    '''
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        # based floor on previous peak
        floor_value = (1-maxdd)*peak_value
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # same as applying min and max
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history


def regress(dependent_variable, explanatory_variables, alpha=True):
    '''
    Runs a linear regression to decompose the dependent variable into the explanatory variables
    returns an object of type statsmodel's RegressionResults on which you can call
        .summay() to print a full summary
        .params for the coefficients
        .tvalues and .pvalues for the significance levels
        .rsquared adj and .rsquared for quality of fit 
    '''
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables['Alpha'] = 1
        
    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm


def tracking_error(r_a, r_b):
    '''
    Returns the tracking error between two return series
    '''
    return np.sqrt(((r_a - r_b)**2).sum())


def portfolio_tracking_error(weights, ref_r, bb_r):
    '''
    returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights
    '''
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))


def style_analysis(dependent_variable, explanatory_variables):
    '''
    Returns the optimal weights that minimizes the tracking error between
    a portfolio of the explanatory variables and the dependent variable
    '''
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    # construct the constraints
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    solution = minimize(portfolio_tracking_error,
                       init_guess,
                       args=(dependent_variable, explanatory_variables,),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    weights = pd.Series(solution.x,
                       index=explanatory_variables.columns)
    return weights


def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    '''
    Returns the weights of the EW portfolio based on the asset returns 'r' as a DataFrame
    If supplied a set of Cap Weights and a Cap Weight tether, it is applied and reweighted
    '''
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]]
        # exclude microcaps
        if microcap_threshold is not None and microcap_threshold > 0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        # limit weight to a multiple of cap weight
        if max_cw_mult is not None and max_cw_mult > 0:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum()
    return ew


def weight_cw(r, cap_weights, **kwargs):
    '''
    Returns the weights of the CW potfolio based on the time series of cap weights
    '''
    return cap_weights.loc[r.index[0]]


def backtest_ws(r, estimation_window=60, weighting=weight_ew, **kwargs):
    '''
    Backtests a given weighting scheme, given some parameters:
        r: asset returns to use to build the portfolio
        estimation_window: the window to use to estimate parameters
        weighting: the weighting scheme to use, must be a function that takes 'r' and a variable number of parameters
    '''
    n_periods = r.shape[0]
    windows = [(start, start+estimation_window) for start in range(n_periods-estimation_window+1)]
    weights = [weighting(r.iloc[win[0]:win[1]], **kwargs) for win in windows]
    # turn list into dataframe
    weights = pd.DataFrame(weights, index=r.iloc[estimation_window-1:].index, columns=r.columns)
    returns = (weights*r).sum(axis=1, min_count=1)
    return returns


def sample_cov(r, **kwargs):
    '''
    Returns the sample covariance of the supplied returns
    '''
    return r.cov()


def cc_cov(r, **kwargs):
    '''
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    '''
    rhos = r.corr()
    n = rhos.shape[0]
    # rhos is a symmetric matrix with diagonals all 1, so mean correlation is:
    rho_bar = (rhos.values.sum()-n)/(n*(n-1))
    ccor = np.full_like(rhos, rho_bar)
    np.fill_diagonal(ccor, 1.)
    sd = r.std()
    ccov = ccor * np.outer(sd, sd)
    return pd.DataFrame(ccov, index=r.columns, columns=r.columns)


def shrinkage_cov(r, delta=0.5, **kwargs):
    '''
    Covariance estimator that shrinks between the sample covariance and the constant correlation estimators
    '''
    prior = cc_cov(r, **kwargs)
    sample = sample_cov(r, **kwargs)
    return delta*prior + (1-delta)*sample


def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    '''
    Produces the weights of the GMV portfolio given a covariance matrix of the returns
    '''
    est_cov = cov_estimator(r, **kwargs)
    return gmv(est_cov)


def as_colvec(x):
    '''
    Given an array of vectors,
    Returns an nd array with axis changed to columns
    '''
    if (x.ndim == 2):
        return x
    else:
        return np.expand_dims(x, axis=1)
    
    
def implied_returns(delta, sigma, w):
    '''
    Obtain the implied expected returns by reverse engineering the weights
    Inputs:
        delta: Risk Aversion Coefficient (scalar)
        sigma: Variance-Covariance Matrix (N x N) as DataFrame
        w: Portfolio weights (N x 1) as Series
    Returns an N x 1 vector of returns as Series
    '''
    ir = delta * sigma.dot(w).squeeze() # squeeze() turns one column dataframe to series
    ir.name = 'Implied Returns'
    return ir


def proportional_prior(sigma, tau, p):
    '''
    Returns the He-Litterman simplified Omega
    Inputs:
        sigma: N x N Covariance Matrix as DataFrame
        tau: a scalar
        p: a K x N DataFrame linking Q and Assets
    Returns a P x P DataFrame, a matrix representing prior uncertainties 
    '''
    # can use .dot() or @ for matrix multiplication
    helit_omega = p.dot(tau * sigma).dot(p.T)
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)), index=p.index, columns=p.index)


def bl(w_prior, sigma_prior, p, q, omega=None, delta=2.5, tau=0.02):
    '''
    Computes the posterior expected returns based on the original black litterman reference model
    Inputs:
        w_prior must be an N x 1 vector of weights, a series
        sigma_prior is an N x N covariance matrix, a DataFrame
        p must be a K x N matrix linking Q and the Assets, a DataFrame
        q must be K x 1 vector of views, a Series
        omega must be a K x K matrix, a DataFrame or None
        if omega is None, we assume it is proportional to variance of the prior
        delta and tau are scalars
    '''
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # number of assets
    N = w_prior.shape[0]
    # number of views
    K = q.shape[0]
    # reverse engineer weights to get pi
    pi = implied_returns(delta, sigma_prior, w_prior)
    # scale sigma by uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior
    # posterior estimate of mean by master formula
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu_bl
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return mu_bl, sigma_bl


def inverse(d):
    '''
    Invert the dataframe by inverting the underlying matrix
    '''
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)


def w_msr(sigma, mu, scale=True):
    '''
    No constrains on this max-sharpe ratio (unlike the previous one we wrote, which is constrained)
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights by using the Markowitz Optimization Procedure
        mu is the vector of excess expected returns
        sigma must be an N x N matrix as a DataFrame and mu a column vector as a series
    '''
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # assumes all w is positive
    return w


def w_star(delta, sigma, mu):
    '''
    Weights in the optimal portfolio, w*
    '''
    return (inverse(sigma).dot(mu))/delta