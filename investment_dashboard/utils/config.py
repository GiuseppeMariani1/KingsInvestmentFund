"""
Configuration settings for King's Investment Fund Dashboard
"""
from datetime import datetime, timedelta

# Data Configuration
DATA_SOURCE = "hybrid"  # "hybrid" (WRDS+Yahoo), "yahoo_finance", or "wrds"
UNIVERSE_SIZE = 600  # Top N stocks by market cap (expanded with WRDS+Yahoo hybrid)
EUROPEAN_MARKET_ONLY = True

# Model Parameters
PLS_LOOKBACK_YEARS = 3
ELASTIC_NET_LOOKBACK_YEARS = 1
PLS_N_COMPONENTS = 6  # Default (CV will select, but minimum 5 for finance: momentum, value, quality, size, vol)
PLS_MIN_COMPONENTS = 5  # Economic minimum (Fama-French + momentum + quality)
ELASTIC_NET_ALPHA = 0.3  # Mixing parameter (0=Ridge, 1=Lasso)

# Portfolio Construction
PORTFOLIO_SIZE = 30  # Number of stocks in portfolio
LONG_ONLY = True
EQUAL_WEIGHT = False  # Use GARCH(1,1) based mean-variance optimization
OPTIMIZATION_METHOD = "garch_mv"  # "garch_mv" (GARCH Mean-Variance), "inverse_vol", "risk_parity"

# Constraints (if using optimization)
MAX_POSITION_SIZE = 0.05  # 5% per stock
MAX_SECTOR_EXPOSURE = 0.25  # 25% per sector
MAX_COUNTRY_EXPOSURE = 0.40  # 40% per country

# Benchmark
BENCHMARK_TICKER = "^STOXX"  # STOXX Europe 600

# Data Update Frequency
RETRAIN_PLS_FREQUENCY = "monthly"  # monthly, quarterly
RETRAIN_ELASTIC_NET_FREQUENCY = "monthly"
REBALANCE_FREQUENCY = "monthly"

# Date Ranges
END_DATE = datetime.now()
PLS_START_DATE = END_DATE - timedelta(days=PLS_LOOKBACK_YEARS * 365)
ELASTIC_NET_START_DATE = END_DATE - timedelta(days=ELASTIC_NET_LOOKBACK_YEARS * 365)

# European Exchanges (for filtering)
EUROPEAN_EXCHANGES = {
    'LSE': 'London Stock Exchange',
    'XETR': 'XETRA (Frankfurt)',
    'XPAR': 'Euronext Paris',
    'MIL': 'Borsa Italiana',
    'AMS': 'Euronext Amsterdam',
    'BRU': 'Euronext Brussels',
    'LIS': 'Euronext Lisbon',
    'MAD': 'Bolsa de Madrid',
    'SWX': 'SIX Swiss Exchange',
    'OSL': 'Oslo Bors',
    'STO': 'Stockholm Stock Exchange',
    'CPH': 'Copenhagen Stock Exchange',
    'HEL': 'Helsinki Stock Exchange',
    'VIE': 'Vienna Stock Exchange',
    'ATH': 'Athens Stock Exchange'
}

# Style Factor Definitions
STYLE_FACTORS = {
    'value': ['pe_ratio', 'pb_ratio', 'ev_ebitda'],
    'momentum': ['return_1m', 'return_3m', 'return_6m', 'return_12m'],
    'quality': ['roe', 'debt_equity', 'profit_margin'],
    'size': ['market_cap'],
    'volatility': ['realized_vol_30d', 'beta'],
    'growth': ['earnings_growth', 'sales_growth']
}

# Macro Factor Tickers (Yahoo Finance) - VERIFIED WORKING TICKERS
MACRO_FACTORS = {
    'interest_rates': {
        'US_10Y': '^TNX',           # US 10Y Treasury yield (proxy for global rates)
        'US_2Y': '^IRX',            # US 3-month T-bill rate
        'EUR_GOVT_BOND': 'IBCI.L',  # iShares EUR Govt Bond 1-3yr ETF
    },
    'volatility': {
        'VIX': '^VIX',              # CBOE Volatility Index
        'VSTOXX': 'FEZ',            # EURO STOXX 50 ETF (proxy for European equity vol)
    },
    'currency': {
        'EURUSD': 'EURUSD=X',
        'EURGBP': 'EURGBP=X',
        'EURCHF': 'EURCHF=X',
        'USDJPY': 'JPY=X',          # USD/JPY for global risk sentiment
    },
    'commodities': {
        'OIL': 'CL=F',              # WTI Crude
        'BRENT': 'BZ=F',            # Brent Crude (more relevant for Europe)
        'GOLD': 'GC=F',
        'COPPER': 'HG=F',           # Industrial metals (economic indicator)
        'NATGAS': 'NG=F',           # Natural Gas (important for Europe)
    },
    'indices': {
        'STOXX600': '^STOXX',
        'STOXX50': '^STOXX50E',     # Euro STOXX 50
        'FTSE100': '^FTSE',
        'CAC40': '^FCHI',
        'DAX': '^GDAXI',
        'SP500': '^GSPC',           # US market for global sentiment
    },
    'credit': {
        'HY_SPREAD': 'HYG',         # High Yield Corporate Bond ETF (credit risk proxy)
        'IG_CORP': 'LQD',           # Investment Grade Corporate Bond ETF
    }
}
