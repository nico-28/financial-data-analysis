import matplotlib.pyplot as plt
import seaborn as sns

def plot_prices_over_time(data):
    """
    Plots the adjusted closing prices of assets over time.

    Parameters:
        data (pd.DataFrame): DataFrame containing the adjusted closing prices of assets.
    """
    plt.figure(figsize=(12, 6))
    data.plot(figsize=(12, 6), title='Stock Prices Over Time')
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.grid()
    plt.show()


def plot_correlation_matrix(daily_returns):
    """
    Plots the correlation matrix as a heatmap.

    Parameters:
        daily_returns (pd.DataFrame): DataFrame of daily returns for assets.
    """
    correlation_matrix = daily_returns.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Matrix - Heatmap")
    plt.show()

def plot_metrics_comparison(metrics_df, title="Comparison of Key Metrics"):
    """
    Generates a bar chart to compare key metrics.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics to compare.
        title (str): Title of the plot.
    """
    metrics_df.plot(kind='bar', figsize=(12, 8), title=title)
    plt.ylabel('Values')
    plt.grid()
    plt.show()

def plot_buy_vs_hold(data, daily_returns, symbol):
    """
    Plots the performance of a Buy-and-Hold strategy vs a strategy based on signals.

    Parameters:
        data (pd.DataFrame): Adjusted close prices of assets.
        daily_returns (pd.DataFrame): Daily returns of assets.
        symbol (str): Stock symbol to analyze.
    """
    prices = data[symbol]
    rolling_mean = prices.rolling(window=20).mean()
    rolling_std = prices.rolling(window=20).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    buy_signals = prices < lower_band
    sell_signals = prices > upper_band

    strategy_returns = daily_returns[symbol].copy()
    strategy_returns[buy_signals] = daily_returns[symbol][buy_signals]
    strategy_returns[sell_signals] = -daily_returns[symbol][sell_signals]
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()
    buy_and_hold_returns = (1 + daily_returns[symbol]).cumprod()

    plt.figure(figsize=(14, 7))
    plt.plot(cumulative_strategy_returns, label='Strategy Returns', alpha=0.7)
    plt.plot(buy_and_hold_returns, label='Buy-and-Hold Returns', alpha=0.7)
    plt.title(f'Strategy vs Buy-and-Hold for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid()
    plt.show()

def plot_optimal_weights(symbols, optimal_weights):
    """
    Plots the optimal portfolio weights as a bar chart.

    Parameters:
        symbols (list): List of asset symbols.
        optimal_weights (list): Optimal weights for the portfolio.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(symbols, optimal_weights, color='skyblue')
    plt.title('Optimal Portfolio Weights')
    plt.ylabel('Weight')
    plt.xlabel('Asset')
    plt.grid()
    plt.show()

def plot_efficient_frontier(mean_returns, cov_matrix, optimal_weights):
    """
    Plots the Efficient Frontier with the optimal portfolio highlighted.

    Parameters:
        mean_returns (array): Expected returns of assets.
        cov_matrix (array): Covariance matrix of asset returns.
        optimal_weights (array): Optimal weights for the portfolio.
    """
    import numpy as np

    portfolio_returns = []
    portfolio_volatilities = []

    num_assets = len(mean_returns)
    for _ in range(5000):
        random_weights = np.random.random(num_assets)
        random_weights /= np.sum(random_weights)

        ret = np.dot(random_weights, mean_returns) * 252
        vol = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights))) * np.sqrt(252)

        portfolio_returns.append(ret)
        portfolio_volatilities.append(vol)

    portfolio_returns = np.array(portfolio_returns)
    portfolio_volatilities = np.array(portfolio_volatilities)

    optimal_return = np.dot(mean_returns, optimal_weights) * 252
    optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights))) * np.sqrt(252)

    plt.figure(figsize=(12, 8))
    plt.scatter(portfolio_volatilities, portfolio_returns, c=(portfolio_returns - 0.04) / portfolio_volatilities, cmap='viridis', marker='o')
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=200, label='Optimal Portfolio')
    plt.title('Efficient Frontier with Optimal Portfolio')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Return')
    plt.legend()
    plt.grid()
    plt.show()

