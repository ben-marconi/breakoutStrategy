# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:28:16 2022

@author: QuantInsti
"""

import numpy as np
import pandas as pd
import empyrical as ep
from tabulate import tabulate
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")

def get_max_drawdown_underwater(underwater):
    """
    Determines peak, valley, and recovery dates given an 'underwater'
    DataFrame.
    An underwater DataFrame is a DataFrame that has precomputed
    rolling drawdown.
    Parameters
    ----------
    underwater : pd.Series
       Underwater returns (rolling drawdown) of a strategy.
    Returns
    -------
    peak : datetime
        The maximum drawdown's peak.
    valley : datetime
        The maximum drawdown's valley.
    recovery : datetime
        The maximum drawdown's recovery.
    """

    valley = underwater.idxmin()  # end of the period
    # Find first 0
    peak = underwater[:valley][underwater[:valley] == 0].index[-1]
    # Find last 0
    try:
        recovery = underwater[valley:][underwater[valley:] == 0].index[0]
    except IndexError:
        recovery = np.nan  # drawdown not recovered
    return peak, valley, recovery

def get_top_drawdowns(returns, top=10):
    """
    Finds top drawdowns, sorted by drawdown amount.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.

    top : int, optional
        The amount of top drawdowns to find (default 10).
    Returns
    -------
    drawdowns : list
        List of drawdown peaks, valleys, and recoveries. See get_max_drawdown.
    """

    returns = returns.copy()
    df_cum = ep.cum_returns(returns, 1.0)
    running_max = np.maximum.accumulate(df_cum)
    underwater = df_cum / running_max - 1

    drawdowns = []
    for _ in range(top):
        peak, valley, recovery = get_max_drawdown_underwater(underwater)
        # Slice out draw-down period
        if not pd.isnull(recovery):
            underwater.drop(underwater[peak: recovery].index[1:-1],
                            inplace=True)
        else:
            # drawdown has not ended yet
            underwater = underwater.loc[:peak]

        drawdowns.append((peak, valley, recovery))
        if ((len(returns) == 0)
                or (len(underwater) == 0)
                or (np.min(underwater) == 0)):
            break

    return drawdowns


def gen_drawdown_table(returns, top=10):
    """
    Places top drawdowns in a table.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    top : int, optional
        The amount of top drawdowns to find (default 10).
    Returns
    -------
    df_drawdowns : pd.DataFrame
        Information about top drawdowns.
    """
    returns = returns.copy()
    df_cum = ep.cum_returns(returns, 1.0)
    drawdown_periods = get_top_drawdowns(returns, top=top)
    df_drawdowns = pd.DataFrame(index=list(range(top)),
                                columns=['Net drawdown in %',
                                         'Peak date',
                                         'Valley date',
                                         'Recovery date',
                                         'Duration'])

    for i, (peak, valley, recovery) in enumerate(drawdown_periods):
        if pd.isnull(recovery):
            df_drawdowns.loc[i, 'Duration'] = np.nan
        else:
            df_drawdowns.loc[i, 'Duration'] = len(pd.date_range(peak,
                                                                recovery,
                                                                freq='B'))
        df_drawdowns.loc[i, 'Peak date'] = (peak.to_pydatetime()
                                            .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Valley date'] = (valley.to_pydatetime()
                                              .strftime('%Y-%m-%d'))
        if isinstance(recovery, float):
            df_drawdowns.loc[i, 'Recovery date'] = recovery
        else:
            df_drawdowns.loc[i, 'Recovery date'] = (recovery.to_pydatetime()
                                                    .strftime('%Y-%m-%d'))
        df_drawdowns.loc[i, 'Net drawdown in %'] = (
            (df_cum.loc[peak] - df_cum.loc[valley]) / df_cum.loc[peak]) * 100

    df_drawdowns['Peak date'] = \
        pd.to_datetime(df_drawdowns['Peak date']).dt.strftime('%Y-%m-%d')
    df_drawdowns['Valley date'] = \
        pd.to_datetime(df_drawdowns['Valley date']).dt.strftime('%Y-%m-%d')
    df_drawdowns['Recovery date'] = \
        pd.to_datetime(df_drawdowns['Recovery date']).dt.strftime('%Y-%m-%d')

    return df_drawdowns


def sharpe_ratio(strategy_returns):
    strategy_returns = strategy_returns.copy()
    strategy_returns.dropna(inplace=True)
    """
    Compute sharpe ratio.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - These daily returns should be provided with a datetime index

    Returns
    -------
    sharpe ratio : A value
        The sharpe ratio of the returns series.
    """
    sharpe = round(strategy_returns.mean() /
          strategy_returns.std() * 252**0.5, 3)
    return sharpe

def cagr(strategy_returns):
    strategy_returns = strategy_returns.copy()
    strategy_returns.dropna(inplace=True)
    """
    Compute the Compounded Annualized Growth ratio (CAGR).
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - These daily returns should be provided with a datetime index

    Returns
    -------
    The CAGR : A value
        The CAGR of the returns series.
    """
    cagr = round(((strategy_returns+1).cumprod().iloc[-1] ** \
                  (252 / len((strategy_returns+1).cumprod()))-1)*100,2)
    return cagr

def mdd(strategy_returns):
    strategy_returns = strategy_returns.copy()
    strategy_returns.dropna(inplace=True)
    """
    Compute the Maximum Drawdown (MDD) of the cumulative strategy_returns.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - These daily returns should be provided with a datetime index

    Returns
    -------
    The MDD : A value
        The MDD of the returns series.
    """
    
    """ Get the Drawdown Table"""
    table1 = gen_drawdown_table(strategy_returns)
    
    """ Get the Maximum Drawdown"""
    mdd = round(table1['Net drawdown in %'].iloc[0],3)
    
    return mdd 

def strategy_stats(strategy_returns):
    """
    Compute strategy statistics.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - These daily returns should be provided with a datetime index

    Returns
    -------
    df_drawdowns : Printed Table
        Information about top drawdowns.
    df_performance_metrics : Printed Table
        Computation of the Sharpe ratio, the CAGR and the MDD
    """

    strategy_returns = strategy_returns.copy()
    strategy_returns.dropna(inplace=True)
    """ Plot the daily returns """
    color1 = '#2ca02c'
    strategy_returns.plot(figsize=(15, 7), color=color1)
    plt.title('Daily Strategy Returns', fontsize=16)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Returns', fontsize=15)
    # Define the tick size for x-axis and y-axis
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()    
    
    """ Plot the cumulative returns """
    color2 = '#1f77b4'
    cumprod_ret = (strategy_returns+1).cumprod()
    cumprod_ret.plot(figsize=(15, 7), color=color2)
    plt.title('Strategy Cumulative Daily Returns', fontsize=16)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Cumulative Returns', fontsize=15)
    # Define the tick size for x-axis and y-axis
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
    """ Plot the Rolling Maximum Drawdown """
    color3 = '#d62728'
    running_max = np.maximum.accumulate(cumprod_ret)
    # Set value of the running_max to 1 if it is less than 1
    running_max[running_max < 1] = 1    
    # Calculate the running maximum drawdown
    running_max_drawdown = (cumprod_ret/running_max)-1
    # Plot the running maximum drawdown
    running_max_drawdown.plot(figsize=(15, 7), color=color3)
    # Define the label for the title of the figure
    plt.title("Drawdown", fontsize=16)
    # Define the labels for x-axis and y-axis
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Drawdown', fontsize=15)
    # Define the tick size for x-axis and y-axis
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)    # Fill the space between the plot
    plt.fill_between(running_max_drawdown.index,
                     running_max_drawdown, alpha=0.5, color=color3, linewidth=0)    
    plt.show()
    
    """ Compute the Drawdown table """
    table1 = gen_drawdown_table(strategy_returns)
    table1 = table1.head(5)
    
    """ Compute our new Sharpe ratio """
    sharpe = sharpe_ratio(strategy_returns)
    
    """ Compute the CAGR """
    cagr_ = cagr(strategy_returns)
    
    """ Maximum Drawdown """
    mdd = str(round(table1['Net drawdown in %'].iloc[0],2))+"%"
    
    table2 = pd.DataFrame({'Metric': ['Sharpe ratio', 'CAGR', 'MDD'],
                          'Value': [sharpe, cagr_, mdd]})
    
    
    # table = pd.DataFrame({'Parameters': ['Sharpe ratio', 'CAGR', 'MDD'],
    #                       'Value': [sharpe, cagr, mdd]})
    
    print(tabulate(table1, headers='keys', tablefmt='psql'))
    print(tabulate(table2, headers='keys', tablefmt='psql'))
