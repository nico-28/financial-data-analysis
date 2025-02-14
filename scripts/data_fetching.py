import yfinance as yf
import pandas as pd

def fetch_data(symbols, start, end):
    """
    Download historical adjusted close prices for a list of symbols.

    Parameters:
    - symbols (list): List of stock symbols.
    - start (str): Start date in 'YYYY-MM-DD' format.
    - end (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: Historical adjusted close prices.
    """
    data = yf.download(symbols, start=start, end=end)['Adj Close']
    return data

def calculate_daily_returns(data):
    """
    Calculate daily returns from adjusted close prices.

    Parameters:
    - data (pd.DataFrame): DataFrame of adjusted close prices.

    Returns:
    - pd.DataFrame: DataFrame of daily returns.
    """
    return data.pct_change().dropna()
