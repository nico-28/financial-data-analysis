# Financial Data Analysis Project

## Overview

This project analyzes financial asset performance, optimizes portfolio allocation, and compares trading strategies. It includes data fetching, exploratory data analysis (EDA), portfolio optimization, and visualization of results.

## Project Structure

```
financial-data-analysis/
│-- main.py    # Main execution script
│-- requirements.txt    # Dependencies file
│-- README.md    # Documentation
│-- LICENSE    # Open-source license
│-- .gitignore    # Files to ignore in Git
│-- /data    # Raw and processed datasets
|   |--price_data.xlsx
│-- /notebooks    # Jupyter Notebooks for exploratory analysis (Currently empty)
│   |-- .gitkeep
│-- /scripts    # Contains scripts for different tasks
│   │-- data_fetching.py  
│   │-- analysis.py  
│   │-- visualization.py  
│   │-- portfolio_optimization.py  
│   │-- __pycache__/   (Automatically generated, should be ignored in Git)
│-- /original_code    # Archive of original scripts before refactoring
│-- /images    # Generated visualizations
│   │-- prices_over_time.png
│   │-- correlation_matrix.png
│   │-- key_metrics_comparison.png
│   │-- strategy_vs_hold.png
│   │-- optimal_portfolio_weights.png
│   │-- efficient_frontier.png

```

## Installation

1. Clone this repository:
    
    ```
    git clone https://github.com/yourusername/financial-data-analysis.git
    cd financial-data-analysis
    
    ```
    
2. Install required dependencies:
    
    ```
    pip install -r requirements.txt
    
    ```
    

## Running the Project

Execute the main script:

```
python main.py

```

This will:

- Fetch financial data
- Perform exploratory data analysis
- Compute portfolio optimization
- Generate key visualizations

## Outputs

### 1. Prices Over Time

![prices_over_time.png](prices_over_time.png)

### 2. Correlation Matrix

![correlation.png](correlation.png)

### 3. Key Metrics Comparison

![key_metrics.png](key_metrics.png)

### 4. Trading Strategy vs Buy & Hold

![strategy1.png](strategy1.png)

![strategy2.png](strategy2.png)

![strategy3.png](strategy3.png)

### 5. Optimal Portfolio Weights

![weights.png](weights.png)

### 6. Efficient Frontier with Optimal Portfolio

![optimal_portfolio.png](optimal_portfolio.png)

## Notes

- Ensure that all dependencies are installed.
- Modify `main.py` to customize asset symbols or analysis parameters.

## License

This project is open-source under the MIT License.

---

🚀 Developed by Nicolas Ramirez