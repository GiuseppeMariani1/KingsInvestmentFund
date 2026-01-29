# King's Investment Fund - Model Framework

## Strategy Overview
A systematic European equity strategy that uses Partial Least Squares (PLS) to compress macro/style factors, then Elastic Net regression to estimate expected returns and select robust stocks for a constrained long-only portfolio.

---

## 1. DATA LAYER

### 1.1 Equity Universe
- **Scope**: Top 500 market cap European stocks
- **Criteria**: 
  - Top 500 by market capitalization
  - European market only (exchanges: LSE, XETR, XPAR, MIL, etc.)
  - Data source: Yahoo Finance
- **Data Requirements**:
  - Stock prices (daily OHLCV)
  - Market cap, sector, country classifications
  - Corporate actions (splits, dividends)

### 1.2 Macro Factors
- **Examples**:
  - Interest rates (EURIBOR, Bund yields)
  - Currency (EUR/USD, EUR/GBP)
  - Commodity prices (oil, metals)
  - Economic indicators (PMI, GDP growth, inflation)
  - Credit spreads (corporate bond spreads)
- **Frequency**: Daily or weekly updates
- **History**: Minimum 3-5 years for model stability

### 1.3 Style Factors
- **Examples**:
  - Value (P/E, P/B, EV/EBITDA)
  - Momentum (1M, 3M, 6M, 12M returns)
  - Quality (ROE, debt/equity, profit margins)
  - Size (market cap)
  - Volatility (realized vol, beta)
  - Growth (earnings growth, sales growth)
- **Frequency**: Daily updates (rolling windows)
- **Calculation**: Stock-level factor exposures

---

## 2. MODEL ARCHITECTURE

### 2.1 Stage 1: Partial Least Squares (PLS)
**Purpose**: Compress highly correlated macro and style factors into a small set of return-relevant components

**Inputs**:
- Matrix X: Macro factors (time series, e.g., 50-100 factors)
- Matrix Y: Stock returns (panel: stocks × time)
- **Note**: PLS finds components that maximize covariance between X and Y

**Outputs**:
- PLS components (typically 5-15 components)
- Component loadings (which factors contribute to each component)
- Component scores (time series of component values)

**Key Parameters**:
- Number of components (cross-validated, typically 5-15)
- Lookback window for training: **3 years** (rolling window)
- Retraining frequency: Monthly
- Use individual stock returns (not aggregated)

### 2.2 Stage 2: Elastic Net Regression
**Purpose**: Estimate expected returns at stock level using PLS components, with regularization to shrink weak/unstable stocks

**Model Specification**:
```
Expected Return_i = α + β₁·PLS₁ + β₂·PLS₂ + ... + βₖ·PLSₖ + εᵢ
```

**Elastic Net Parameters**:
- **α (alpha)**: Mixing parameter (0 = Ridge, 1 = Lasso)
  - Typical range: 0.1-0.5 (balances L1/L2 regularization)
- **λ (lambda)**: Regularization strength
  - Determined by cross-validation
  - Controls shrinkage: higher λ → fewer stocks with non-zero coefficients

**Outputs**:
- Expected return for each stock
- Model coefficients (β values)
- Model fit metrics (R², cross-validation error)

**Key Parameters**:
- Lookback window for training: **1 year** (rolling window)
- α (mixing parameter): 0.3 (default, can be tuned)
- Cross-validation: Time-series CV to avoid lookahead bias
- Retraining frequency: Monthly

### 2.3 Stage 3: Stock Ranking
**Purpose**: Rank stocks by model-implied expected return

**Process**:
1. Sort stocks by expected return (descending)
2. Filter out stocks with negative expected returns? (long-only constraint)
3. Apply additional filters if needed (liquidity, market cap, etc.)

**Output**: Ranked list of stocks with expected returns

---

## 3. PORTFOLIO CONSTRUCTION

### 3.1 Constraints
**Long-Only**:
- All weights ≥ 0 (no shorting)

**Other Potential Constraints**:
- Maximum position size (e.g., 5% per stock)
- Maximum sector exposure (e.g., 25% per sector)
- Maximum country exposure (e.g., 40% per country)
- Turnover limits (e.g., max 20% monthly turnover)
- Minimum position size (e.g., 0.5% to avoid too many tiny positions)

**Questions to Clarify**:
- How many stocks in portfolio? (Top N, or all above threshold?)
- What constraints beyond long-only?
- Equal-weighted top N, or optimize weights?

### 3.2 Optimization Method
**Option A: Simple Top-N Equal Weight**
- Select top N stocks by expected return
- Equal weight each (1/N)

**Option B: Constrained Optimization**
- Objective: Maximize expected return (or Sharpe ratio)
- Constraints: Long-only, position limits, sector/country limits
- Solver: Quadratic programming (e.g., `cvxpy`, `scipy.optimize`)

**Recommendation**: Start with Option A, add optimization later if needed

---

## 4. DASHBOARD COMPONENTS

### 4.1 Model Overview
- Current model date/version
- Number of stocks in universe
- Number of factors (macro + style)
- Number of PLS components
- Model fit metrics (R², cross-validation error)

### 4.2 Factor Analysis
- **PLS Component Loadings**: Which factors drive each component
- **Component Time Series**: Historical values of PLS components
- **Factor Importance**: Contribution of each factor to expected returns

### 4.3 Stock Rankings
- **Ranked List**: Top stocks by expected return
- **Expected Returns**: Model-implied returns for each stock
- **Factor Exposures**: Each stock's exposure to PLS components
- **Historical Performance**: Past returns, volatility, Sharpe ratio

### 4.4 Portfolio View
- **Current Holdings**: Stocks, weights, expected returns
- **Sector/Country Breakdown**: Portfolio composition
- **Risk Metrics**: Portfolio volatility, beta, concentration
- **Turnover**: Changes from previous period

### 4.5 Performance Tracking
- **Backtest Results**: Historical performance vs. benchmark (e.g., STOXX 600)
- **Metrics**: 
  - Total return, Sharpe ratio, max drawdown
  - Alpha, beta, tracking error
  - Win rate, average holding period
- **Attribution**: Factor vs. stock selection contribution

### 4.6 Model Diagnostics
- **Residual Analysis**: Model errors over time
- **Stability**: Coefficient stability across time periods
- **Out-of-Sample Performance**: Validation set performance

---

## 5. OPERATIONAL DETAILS

### 5.1 Model Retraining Frequency
- **PLS**: Monthly or quarterly (macro factors change slowly)
- **Elastic Net**: Monthly (more frequent to capture changing relationships)
- **Portfolio Rebalancing**: Monthly (aligned with model updates)

### 5.2 Data Updates
- **Daily**: Stock prices, returns
- **Weekly**: Macro factors (if available)
- **Monthly**: Style factors (rolling calculations), full model retrain

### 5.3 Benchmark
- **Primary**: STOXX Europe 600 (broad European index)
- **Secondary**: Sector indices for attribution

---

## 6. OPEN QUESTIONS TO RESOLVE

### Data & Universe
1. **Data Sources**: ✅ **Yahoo Finance**
2. **Universe Definition**: ✅ **Top 500 market cap European stocks**

### Model Parameters
3. **PLS Components**: 
   - How many components? (Cross-validate or fixed?)
   - Should we aggregate stocks to sector level for PLS, or use individual stocks?
4. **Elastic Net**:
   - What α (mixing parameter)? (0.2-0.5 typical)
   - Cross-validation scheme? (Time-series CV)
5. **Lookback Windows**: ✅ **RESOLVED**
   - PLS training: **3 years**
   - Elastic Net training: **1 year**

### Portfolio Construction
6. **Portfolio Size**: 
   - How many stocks? (20-50 typical for concentrated portfolios)
   - Top N by expected return, or all above threshold?
7. **Constraints**:
   - Beyond long-only, what limits? (Position size, sector, country, turnover)
8. **Weighting**:
   - Equal weight top N, or optimize weights?

### Dashboard & Visualization
9. **Key Metrics**: 
   - What are the most important KPIs to display?
10. **Update Frequency**: 
    - Real-time, daily, weekly?
    - How often should users see new rankings?

---

## 7. PROPOSED IMPLEMENTATION STRUCTURE

```
investment_dashboard/
├── data/
│   ├── equity_universe.py      # Stock universe definition
│   ├── macro_factors.py         # Macro factor data loading
│   ├── style_factors.py        # Style factor calculation
│   └── data_loader.py          # Unified data interface
├── models/
│   ├── pls_model.py            # PLS component extraction
│   ├── elastic_net_model.py    # Elastic Net regression
│   └── model_pipeline.py       # End-to-end model execution
├── portfolio/
│   ├── ranking.py              # Stock ranking logic
│   ├── optimization.py         # Portfolio construction
│   └── constraints.py          # Constraint definitions
├── dashboard/
│   ├── app.py                  # Main Streamlit app
│   ├── pages/
│   │   ├── overview.py         # Model overview
│   │   ├── factors.py          # Factor analysis
│   │   ├── rankings.py         # Stock rankings
│   │   ├── portfolio.py        # Portfolio view
│   │   └── performance.py      # Performance tracking
│   └── components/             # Reusable UI components
├── backtesting/
│   ├── backtest_engine.py      # Historical simulation
│   └── performance_metrics.py  # Performance calculation
└── utils/
    ├── config.py               # Configuration settings
    └── helpers.py              # Utility functions
```

---

## NEXT STEPS

1. **Clarify open questions** (Section 6) - especially data sources and parameters
2. **Define data schema** - exact fields and formats
3. **Set up data pipeline** - how to fetch/update data
4. **Implement model components** - PLS → Elastic Net → Ranking
5. **Build dashboard** - Streamlit interface with visualizations
6. **Add backtesting** - Historical performance validation
7. **Deploy & monitor** - Production deployment and monitoring

---

**Please review this framework and let me know:**
- Which questions from Section 6 need answers?
- Any changes to the model architecture?
- Additional dashboard components needed?
- Any constraints or requirements I missed?

Once we align on the framework, we can proceed with implementation.
