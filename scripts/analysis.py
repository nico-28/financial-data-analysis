import numpy as np

def calculate_sharpe_ratio(returns, risk_free_rate=0.04):
    """
    Calculate the Sharpe Ratio for a given set of daily returns.

    Parameters:
    - returns (pd.Series): Daily returns of the asset.
    - risk_free_rate (float): Annual risk-free rate (default: 0.04).

    Returns:
    - float: Sharpe Ratio.
    """
    return (returns.mean() - risk_free_rate / 252) / returns.std() * (252 ** 0.5)

def calculate_sortino_ratio(returns, risk_free_rate=0.04):
    """
    Calculate the Sortino Ratio for a given set of daily returns.

    Parameters:
    - returns (pd.Series): Daily returns of the asset.
    - risk_free_rate (float): Annual risk-free rate (default: 0.04).

    Returns:
    - float: Sortino Ratio.
    """
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    return (returns.mean() - risk_free_rate / 252) / downside_std * (252 ** 0.5)

def calculate_max_drawdown(cumulative_returns):
    """
    Calculate the maximum drawdown from cumulative returns.

    Parameters:
    - cumulative_returns (pd.Series): Cumulative returns.

    Returns:
    - float: Maximum drawdown.
    """
    roll_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / roll_max - 1
    return drawdown.min()
