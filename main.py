from scripts.data_fetching import fetch_data, calculate_daily_returns
from scripts.analysis import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown
from scripts.visualization import (
    plot_prices_over_time,
    plot_correlation_matrix,
    plot_metrics_comparison,
    plot_buy_vs_hold,
    plot_optimal_weights,
    plot_efficient_frontier
)
from scripts.portfolio_optimization import optimize_portfolio
import pandas as pd

# Step 1: Fetch data
symbols = ['AAPL', 'MSFT', 'GOOGL']
data = fetch_data(symbols, start='2020-01-01', end='2023-01-01')
daily_returns = calculate_daily_returns(data)

# Step 2: Visualize Prices Over Time
plot_prices_over_time(data)

# Step 3: Visualize Correlation Matrix
plot_correlation_matrix(daily_returns)

# Step 4: Calculate financial metrics
sharpe_ratios = daily_returns.apply(calculate_sharpe_ratio)
sortino_ratios = daily_returns.apply(calculate_sortino_ratio)
max_drawdowns = daily_returns.apply(lambda x: calculate_max_drawdown((1 + x).cumprod()))

# Print metrics
print("Sharpe Ratios:\n", sharpe_ratios)
print("Sortino Ratios:\n", sortino_ratios)
print("Maximum Drawdowns:\n", max_drawdowns)

# Step 5: Visualize Key Metrics
metrics_df = pd.DataFrame({
    'Sharpe Ratio': sharpe_ratios,
    'Sortino Ratio': sortino_ratios,
    'Maximum Drawdown': max_drawdowns
})
plot_metrics_comparison(metrics_df)

# Step 6: Visualize Strategy vs Buy & Hold
for symbol in symbols:
    plot_buy_vs_hold(data, daily_returns, symbol)

# Step 7: Optimize Portfolio
mean_returns = daily_returns.mean().values
cov_matrix = daily_returns.cov().values
optimal_weights = optimize_portfolio(mean_returns, cov_matrix)

# Print optimal weights
print("Optimal Portfolio Weights:")
for symbol, weight in zip(symbols, optimal_weights):
    print(f"{symbol}: {weight:.2%}")

# Visualize Optimal Portfolio Weights
plot_optimal_weights(symbols, optimal_weights)

# Step 8: Visualize Efficient Frontier
plot_efficient_frontier(mean_returns, cov_matrix, optimal_weights)
