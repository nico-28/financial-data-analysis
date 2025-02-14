import numpy as np
import cvxpy as cp

def optimize_portfolio(mean_returns, cov_matrix, target_return=0.001):
    """
    Optimize the portfolio allocation to minimize volatility.

    Parameters:
    - mean_returns (np.array): Expected returns for each asset.
    - cov_matrix (np.array): Covariance matrix of asset returns.
    - target_return (float): Desired minimum portfolio return (default: 0.001).

    Returns:
    - np.array: Optimal weights for each asset.
    """
    num_assets = len(mean_returns)
    weights = cp.Variable(num_assets)  # Portfolio weights are variables to optimize
    portfolio_return = mean_returns @ weights
    portfolio_volatility = cp.quad_form(weights, cov_matrix)
    
    # Define the optimization objective and constraints
    objective = cp.Minimize(portfolio_volatility)
    constraints = [
        cp.sum(weights) == 1,  # Fully invested
        weights >= 0,          # No short selling
        portfolio_return >= target_return  # Meet the target return
    ]
    
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return weights.value
