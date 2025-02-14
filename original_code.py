import pandas as pd
import yfinance as yf
import seaborn as sns
import numpy as np
import cvxpy as cp

# Define stock symbols
symbols = ['AAPL', 'MSFT', 'GOOGL']

# Download historical data
data = yf.download(symbols, start='2020-01-01', end='2023-01-01')['Adj Close']

# Save the data to a CSV file (using your specified directory)
file_path = 'C:/Users/HP/OneDrive/Escritorio/financial-data-analysis/data/price_data.csv'
data.to_csv(file_path)

print("Data downloaded:")
print(data.head())
print(f"File saved at: {file_path}")

print("Data Summary:")
print(data.describe())

print("Checking for missing values:")
print(data.isnull().sum())

print("Data types:")
print(data.dtypes)

# Plotting the data
import matplotlib.pyplot as plt

data.plot(figsize=(10, 6), title='Stock Prices Over Time')
plt.xlabel("Date")
plt.ylabel("Adjusted Close Price")
plt.grid()
plt.show()

# Calculate daily return 
daily_returns = data.pct_change().dropna()
print("Daily returns:")
print(daily_returns.head())

# Calculate and visualize the correlation matrix
correlation_matrix = daily_returns.corr()
print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize = (8,6))
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidths = 0.5)
plt.title("correlation matrix - heat map")
plt.show()

# Calculate cumulative returns

cumulative_returns = (1 + daily_returns).cumprod()
print("Cumulative returns:")
print(cumulative_returns.head())

cumulative_returns.plot(figsize = (10, 6 ), title = 'Cumulative returns over time')
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid()
plt.show()

#calculated annualized volatility
annualized_volatility = daily_returns.std() * (252 ** 0.5)
print("Annualized volatility:")
print(annualized_volatility)

# Visualize the distribution of returns
daily_returns.plot(kind='hist', bins=50, alpha=0.7, figsize=(10, 6), title='Distribution of Daily Returns')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# Advanced Metrics
rf_rate = 0.04 / 252  # Assuming a 4% annual risk-free rate divided by 252 trading days

# Sharpe Ratio
sharpe_ratios = (daily_returns.mean() - rf_rate) / daily_returns.std() * (252 ** 0.5)
print("Sharpe Ratios:")
print(sharpe_ratios)

# Maximum Drawdown
def calculate_max_drawdown(cumulative_returns):
    roll_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / roll_max - 1
    max_drawdown = drawdown.min()
    return max_drawdown

max_drawdowns = cumulative_returns.apply(calculate_max_drawdown)
print("Maximum Drawdowns:")
print(max_drawdowns)

# Sortino Ratio
def calculate_sortino_ratio(returns, rf_rate):
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino = (returns.mean() - rf_rate) / downside_std * (252 ** 0.5)
    return sortino

sortino_ratios = daily_returns.apply(calculate_sortino_ratio, rf_rate=rf_rate)
print("Sortino Ratios:")
print(sortino_ratios)

# Beta Calculation
market_symbol = 'SPY'  # Assuming SPY represents the market
market_data = yf.download(market_symbol, start='2020-01-01', end='2023-01-01')['Adj Close']
market_returns = market_data.pct_change().dropna()

betas = {}
for symbol in symbols:
    cov_matrix = np.cov(daily_returns[symbol], market_returns)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    betas[symbol] = beta

print("Betas:")
print(betas)

# Alpha Calculation
def calculate_alpha(asset_returns, market_returns, beta, rf_rate):
    excess_market_return = market_returns.mean() - rf_rate
    alpha = asset_returns.mean() - (rf_rate + beta * excess_market_return)
    return alpha

alphas = {}
for symbol in symbols:
    alphas[symbol] = calculate_alpha(daily_returns[symbol], market_returns, betas[symbol], rf_rate)

print("Alphas:")
print(alphas)

# Value at Risk (VaR)
def calculate_var(returns, confidence_level=0.05):
    return np.percentile(returns, confidence_level * 100)

var_values = daily_returns.apply(calculate_var)
print("Value at Risk (5%):")
print(var_values)

# Advanced Visualizations
# 1. Comparison of Key Metrics
metrics_df = pd.DataFrame({
    'Sharpe Ratio': sharpe_ratios,
    'Sortino Ratio': sortino_ratios,
    'Maximum Drawdown': max_drawdowns,
    'Beta': pd.Series(betas)
})

metrics_df.plot(kind='bar', figsize=(12, 8), title='Comparison of Key Metrics')
plt.ylabel('Values')
plt.grid()
plt.show()

# 2. Moving Averages and Bollinger Bands
for symbol in symbols:
    plt.figure(figsize=(12, 8))
    data[symbol].plot(label='Price', alpha=0.7)
    data[symbol].rolling(window=20).mean().plot(label='20-Day SMA')
    rolling_std = data[symbol].rolling(window=20).std()
    (data[symbol].rolling(window=20).mean() + (2 * rolling_std)).plot(label='Upper Bollinger Band', linestyle='--')
    (data[symbol].rolling(window=20).mean() - (2 * rolling_std)).plot(label='Lower Bollinger Band', linestyle='--')
    plt.title(f'{symbol} - Price, Moving Averages, and Bollinger Bands')
    plt.legend()
    plt.grid()
    plt.show()
    
# Segment Analysis
# 1. Monthly Returns
daily_returns['Month'] = daily_returns.index.month
daily_returns['Year'] = daily_returns.index.year

monthly_returns = daily_returns.groupby(['Year', 'Month']).mean()
print("Monthly Returns:")
print(monthly_returns.head())

monthly_returns.unstack(level=0).plot(kind='bar', subplots=True, figsize=(15, 10), title='Monthly Average Returns by Year')
plt.xlabel('Month')
plt.ylabel('Average Daily Return')
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid()
plt.show()

# 2. Annual Returns
annual_returns = daily_returns.groupby('Year').mean()
print("Annual Returns:")
print(annual_returns.head())

annual_returns.plot(kind='bar', figsize=(10, 6), title='Annual Average Returns')
plt.xlabel('Year')
plt.ylabel('Average Daily Return')
plt.grid()
plt.show()

# 3. Maximum and Minimum Prices
max_prices = data.max()
min_prices = data.min()
print("Maximum Prices:")
print(max_prices)
print("Minimum Prices:")
print(min_prices)

plt.figure(figsize=(12, 6))
data.plot(title='Stock Prices with Max and Min Highlights')
for symbol in symbols:
    plt.scatter(data[data[symbol] == max_prices[symbol]].index, max_prices[symbol], label=f'{symbol} Max', color='green')
    plt.scatter(data[data[symbol] == min_prices[symbol]].index, min_prices[symbol], label=f'{symbol} Min', color='red')
plt.legend()
plt.grid()
plt.show()

# Backtesting with Bollinger Bands
for symbol in symbols:
    prices = data[symbol]
    rolling_mean = prices.rolling(window=20).mean()
    rolling_std = prices.rolling(window=20).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)

    # Buy and sell signals
    buy_signals = prices < lower_band
    sell_signals = prices > upper_band

    # Plot signals
    plt.figure(figsize=(14, 7))
    plt.plot(prices, label='Price', alpha=0.7)
    plt.plot(rolling_mean, label='20-Day SMA', linestyle='--')
    plt.plot(upper_band, label='Upper Band', linestyle='--')
    plt.plot(lower_band, label='Lower Band', linestyle='--')
    plt.scatter(prices[buy_signals].index, prices[buy_signals], label='Buy Signal', marker='^', color='green', alpha=1)
    plt.scatter(prices[sell_signals].index, prices[sell_signals], label='Sell Signal', marker='v', color='red', alpha=1)
    plt.title(f'Bollinger Bands with Buy/Sell Signals for {symbol}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    # Strategy performance
    strategy_returns = daily_returns[symbol].copy()
    strategy_returns[buy_signals] = daily_returns[symbol][buy_signals]
    strategy_returns[sell_signals] = -daily_returns[symbol][sell_signals]
    cumulative_strategy_returns = (1 + strategy_returns).cumprod()

    # Compare with buy-and-hold
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
    
# Portfolio Optimization
mean_returns = np.array(daily_returns[symbols].mean())  # Convert to Numpy array
cov_matrix = daily_returns[symbols].cov().values  # Convert to Numpy array
num_assets = len(symbols)
rf_rate = 0.04 / 252  # Assuming a 4% annual risk-free rate divided by 252 trading days

# Variables for optimization
weights = cp.Variable(num_assets)  # Define weights as a 1D vector
portfolio_return = mean_returns @ weights  # Dot product for weighted return
portfolio_volatility = cp.quad_form(weights, cov_matrix)  # Quadratic form for volatility

# Set a target return for the portfolio
target_return = 0.001  # Example: 0.1% daily return

# Objective: Minimize portfolio volatility
objective = cp.Minimize(portfolio_volatility)
constraints = [
    cp.sum(weights) == 1,  # Fully invested
    weights >= 0,          # No short selling
    portfolio_return >= target_return  # Achieve at least the target return
]

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Optimal weights
optimal_weights = weights.value
print("Optimal Weights:")
for symbol, weight in zip(symbols, optimal_weights):
    print(f"{symbol}: {weight:.2%}")

# Portfolio performance
optimal_return = np.dot(mean_returns, optimal_weights) * 252  # Annualized return
optimal_volatility = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights) * np.sqrt(252)  # Annualized volatility
optimal_sharpe = (optimal_return - 0.04) / optimal_volatility

print("\nOptimized Portfolio Performance:")
print(f"Annualized Return: {optimal_return:.2%}")
print(f"Annualized Volatility: {optimal_volatility:.2%}")
print(f"Sharpe Ratio: {optimal_sharpe:.2f}")



# Visualize optimal weights
plt.figure(figsize=(10, 6))
plt.bar(symbols, optimal_weights, color='skyblue')
plt.title('Optimal Portfolio Weights')
plt.ylabel('Weight')
plt.xlabel('Asset')
plt.grid()
plt.show()

# Efficient Frontier
risk_free_rate = 0.04
portfolio_returns = []
portfolio_volatilities = []
weights_record = []

for _ in range(5000):
    random_weights = np.random.random(num_assets)
    random_weights /= np.sum(random_weights)
    weights_record.append(random_weights)

    ret = np.dot(random_weights, mean_returns.flatten()) * 252
    vol = np.sqrt(np.dot(random_weights.T, np.dot(cov_matrix, random_weights))) * np.sqrt(252)

    portfolio_returns.append(ret)
    portfolio_volatilities.append(vol)

portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)

plt.figure(figsize=(12, 8))
plt.scatter(portfolio_volatilities, portfolio_returns, c=(portfolio_returns - risk_free_rate) / portfolio_volatilities, cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(optimal_volatility, optimal_return, color='red', marker='*', s=200, label='Optimal Portfolio')
plt.title('Efficient Frontier with Optimal Portfolio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Return')
plt.legend()
plt.grid()
plt.show()
































