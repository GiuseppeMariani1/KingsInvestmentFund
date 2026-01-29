# King's Investment Fund Dashboard

A systematic European equity investment strategy dashboard using Partial Least Squares (PLS) and Elastic Net regression.

## Strategy Overview

1. **Data Layer**: Top 500 European stocks + macro/style factors (Yahoo Finance)
2. **PLS Model**: Compresses factors into return-relevant components (3-year lookback)
3. **Elastic Net**: Estimates expected returns with regularization (1-year lookback)
4. **Portfolio Construction**: Long-only portfolio from top-ranked stocks

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit dashboard:

```bash
streamlit run investment_dashboard/dashboard/app.py
```

Or from the project root:

```bash
cd investment_dashboard/dashboard
streamlit run app.py
```

## Project Structure

```
investment_dashboard/
├── data/              # Data loading modules
├── models/            # PLS and Elastic Net models
├── portfolio/         # Portfolio construction
├── dashboard/         # Streamlit dashboard
├── backtesting/       # Performance tracking (future)
└── utils/             # Configuration and utilities
```

## Configuration

Edit `investment_dashboard/utils/config.py` to adjust:
- Universe size
- Model parameters
- Portfolio constraints
- Data sources

## Model Parameters

- **PLS Lookback**: 3 years
- **Elastic Net Lookback**: 1 year
- **Portfolio Size**: Top 30 stocks (configurable)
- **Weighting**: Equal-weighted (can be extended to optimization)

## Notes

- Data is fetched from Yahoo Finance (free tier)
- Model retraining frequency: Monthly
- Portfolio rebalancing: Monthly
- Benchmark: STOXX Europe 600

## Future Enhancements

- [ ] Backtesting engine
- [ ] Performance attribution
- [ ] Risk metrics (VaR, CVaR)
- [ ] Portfolio optimization (mean-variance, risk parity)
- [ ] Real-time data updates
- [ ] Export functionality
