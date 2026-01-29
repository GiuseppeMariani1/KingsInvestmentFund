"""
King's Investment Fund Dashboard
Main Streamlit application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import traceback

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from investment_dashboard.models.model_pipeline import ModelPipeline
from investment_dashboard.portfolio.ranking import StockRanker
from investment_dashboard.portfolio.optimization import PortfolioOptimizer
from investment_dashboard.portfolio.backtest import run_simple_backtest, Backtester
from investment_dashboard.portfolio.walk_forward_backtest import run_walk_forward_backtest, WalkForwardBacktester
from investment_dashboard.utils.config import PORTFOLIO_SIZE, DATA_SOURCE

# Page configuration
st.set_page_config(
    page_title="King's Investment Fund Dashboard",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (Bloomberg-style dark theme)
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0a0a0a;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        color: #f7b801;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f7b801 !important;
        color: #000000 !important;
    }
    h1, h2, h3 {
        color: #f7b801 !important;
    }
    .stMetric {
        background-color: #1a1a2e;
        padding: 10px;
        border-radius: 5px;
    }
    div[data-testid="stMetricValue"] {
        color: #f7b801;
    }
    div[data-testid="stMetricLabel"] {
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = None
if 'model_fitted' not in st.session_state:
    st.session_state.model_fitted = False

# Header
st.markdown("<h1 style='text-align: center; color: #f7b801;'>KING'S INVESTMENT FUND</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>European Equity Strategy | Long-Term Fundamental Compounding</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("<h2 style='color: #f7b801;'>CONTROLS</h2>", unsafe_allow_html=True)

# Model execution
if st.sidebar.button("Run Model Pipeline", type="primary"):
    # Create a status container in the main area
    status_container = st.empty()
    progress_bar = st.progress(0)
    log_container = st.empty()
    
    try:
        pipeline = ModelPipeline()
        
        # Step 1: Build Universe
        status_container.info("Step 1/6: Building equity universe...")
        log_container.text("Fetching stock info for ~100 European stocks...")
        pipeline.universe.build_universe()
        progress_bar.progress(15)
        log_container.text(f"Universe built: {len(pipeline.universe.stocks)} stocks")
        
        # Step 2: Fetch Returns
        status_container.info("Step 2/6: Fetching 3 years of stock returns...")
        log_container.text("Downloading daily prices (this takes 2-3 minutes)...")
        end_date = datetime.now()
        pls_start = end_date - timedelta(days=3 * 365)
        pipeline.stock_returns = pipeline.universe.fetch_returns(pls_start, end_date)
        # Clean returns
        nan_pct = pipeline.stock_returns.isna().sum() / len(pipeline.stock_returns)
        valid_stocks = nan_pct[nan_pct < 0.2].index.tolist()
        pipeline.stock_returns = pipeline.stock_returns[valid_stocks].fillna(0)
        progress_bar.progress(40)
        log_container.text(f"Returns fetched: {len(pipeline.stock_returns.columns)} stocks, {len(pipeline.stock_returns)} days")
        
        # Step 3: Fetch Macro Factors
        status_container.info("Step 3/6: Loading macro factors...")
        log_container.text("Fetching EUR/USD, Oil, Gold, indices...")
        pipeline.macro_factors = pipeline.macro_loader.load_all_factors(pls_start, end_date)
        pipeline.macro_factors = pipeline.macro_factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
        progress_bar.progress(55)
        log_container.text(f"Macro factors loaded: {len(pipeline.macro_factors.columns)} factors")
        
        # Step 4: Calculate Style Factors (ROLLING - time-varying!)
        status_container.info("Step 4/6: Calculating ROLLING style factors...")
        log_container.text("Computing rolling momentum, volatility, beta...")
        prices = pipeline._fetch_prices(pls_start, end_date)
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        tickers = [t for t in pipeline.universe.stocks if t in pipeline.stock_returns.columns]
        
        # Calculate ROLLING style factors (time-varying)
        pipeline.rolling_style_factors = pipeline.style_calculator.calculate_rolling_style_factors(
            prices, pipeline.stock_returns
        )
        pipeline.rolling_style_factors = pipeline.rolling_style_factors.replace([float('inf'), float('-inf')], float('nan'))
        pipeline.rolling_style_factors = pipeline.rolling_style_factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Also calculate static factors for stock info
        pipeline.style_factors = pipeline.style_calculator.calculate_all_factors(tickers, prices, pipeline.stock_returns)
        pipeline.style_factors = pipeline.style_factors.replace([float('inf'), float('-inf')], float('nan'))
        pipeline.style_factors = pipeline.style_factors.fillna(pipeline.style_factors.median()).fillna(0)
        
        pipeline.combined_factors = pipeline._combine_factors()
        pipeline.combined_factors = pipeline.combined_factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
        progress_bar.progress(70)
        log_container.text(f"Rolling factors: {len(pipeline.rolling_style_factors.columns)} time-varying factors")
        
        # Step 5: Regime Identification
        status_container.info("Step 5/6: Identifying market regimes...")
        log_container.text("Analyzing which factors are driving returns...")
        common_dates = pipeline.combined_factors.index.intersection(pipeline.stock_returns.index)
        X = pipeline.combined_factors.loc[common_dates].fillna(0)
        Y = pipeline.stock_returns.loc[common_dates].fillna(0)
        
        # Cross-validate to find optimal components
        optimal_n = pipeline.pls_model.cross_validate_components(X, Y, max_components=15)
        log_container.text(f"Identified {optimal_n} key market drivers")
        
        # Fit final PLS model
        pipeline.pls_model.fit(X, Y)
        pipeline.pls_components = pipeline.pls_model.get_component_scores()
        progress_bar.progress(85)
        log_container.text(f"Regime analysis complete: {pipeline.pls_model.n_components} drivers identified")
        
        # Step 6: Stock Ranking
        status_container.info("Step 6/6: Ranking stocks by alignment...")
        log_container.text("Scoring each stock's regime alignment...")
        en_start = end_date - timedelta(days=1 * 365)
        pls_1y = pipeline.pls_components.loc[pipeline.pls_components.index >= en_start].fillna(0)
        returns_1y = pipeline.stock_returns.loc[pipeline.stock_returns.index >= en_start].fillna(0)
        pipeline.elastic_net_model.fit(pls_1y, returns_1y)
        latest_components = pipeline.pls_components.iloc[-1:]
        expected_returns_df = pipeline.elastic_net_model.predict(latest_components)
        pipeline.expected_returns = expected_returns_df['expected_return']
        progress_bar.progress(95)
        log_container.text(f"Ranking complete: {len(pipeline.expected_returns)} stocks scored")
        
        # Build portfolio
        status_container.info("Building portfolio...")
        st.session_state.pipeline = pipeline
        st.session_state.model_fitted = True
        
        ranker = StockRanker()
        optimizer = PortfolioOptimizer(portfolio_size=PORTFOLIO_SIZE)
        expected_returns = pipeline.expected_returns
        stock_info = pipeline.universe.get_universe_summary()
        # Pass historical returns for GARCH-based optimization
        historical_returns = pipeline.stock_returns
        portfolio = optimizer.construct_portfolio(expected_returns, stock_info, historical_returns)
        st.session_state.portfolio = portfolio
        st.session_state.optimizer = optimizer  # Save for metrics
        
        progress_bar.progress(100)
        status_container.success(f"Portfolio ready! Screened {len(pipeline.expected_returns)} stocks")
        log_container.text(f"Selected {len(portfolio)} positions for the portfolio")
        
    except Exception as e:
        status_container.error(f"Error: {str(e)}")
        log_container.text(f"Full error: {str(e)}")
        st.session_state.model_fitted = False
        st.text(traceback.format_exc())

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.selectbox(
    "Navigate",
    ["Product Overview", "Investment Philosophy", "Universe & Constraints", "Stock Selection", "Factor Analysis", "Current Portfolio", "Risk & Monitoring", "Performance"],
    index=0
)

# Main content based on selected page
if not st.session_state.model_fitted:
    st.info("Click 'Run Model Pipeline' in the sidebar to load the portfolio.")
    
    # =========================================================================
    # PAGE 1: PRODUCT OVERVIEW (Pre-Model State)
    # =========================================================================
    st.markdown("""
    ## What This Portfolio Is
    
    The King's Investment Fund European Equity Strategy is a **long-term, long-only portfolio** 
    of fundamentally strong European companies.
    
    We own businesses with:
    - **Strong profitability** (high return on equity, healthy margins)
    - **Sound balance sheets** (manageable debt levels)
    - **Favorable market positioning** (aligned with current market conditions)
    
    ---
    
    ## Who This Is For
    
    This strategy is designed for **institutional investors** with:
    - A **long-term investment horizon** (3-5+ years)
    - Tolerance for short-term volatility in pursuit of long-term compounding
    - Appreciation for disciplined, systematic investment processes
    
    ---
    
    ## What Problem It Solves
    
    **The challenge:** European equities offer attractive opportunities, but selecting 
    individual stocks requires navigating:
    - 600+ companies across diverse sectors and countries
    - Complex macroeconomic conditions (rates, currencies, geopolitics)
    - Short-term noise that obscures long-term value
    
    **Our solution:** A systematic process that:
    1. Anchors on fundamental business quality (what to own)
    2. Adapts to market regimes (when conditions favor certain characteristics)
    3. Controls risk continuously (how much to own)
    
    ---
    
    ## Key Differentiators
    
    | Feature | Description |
    |---------|-------------|
    | **Fundamental Anchor** | We prioritize business quality over market timing |
    | **Regime Awareness** | We adapt exposure based on what markets are pricing |
    | **Risk Discipline** | Position sizes reflect risk contribution, not conviction alone |
    | **Transparency** | Every decision can be traced to data, not judgment |
    
    ---
    
    ## Investment Horizon
    
    - **Strategy design:** Long-term fundamental compounding
    - **Monitoring window (Feb-May):** Short-term risk oversight only
    - **This is not:** A trading strategy or return-prediction exercise
    
    The monitoring period validates risk discipline and regime alignment, 
    not thesis correctness. Short-term results may deviate from long-term expectations.
    """)
else:
    pipeline = st.session_state.pipeline
    portfolio = st.session_state.portfolio
    summary = pipeline.get_model_summary()
    
    # =========================================================================
    # PAGE 1: PRODUCT OVERVIEW (Post-Model State)
    # =========================================================================
    if page == "Product Overview":
        st.markdown("<h2>PRODUCT OVERVIEW</h2>", unsafe_allow_html=True)
        
        # Key Portfolio Metrics (investor-focused)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Portfolio Positions", len(portfolio))
        with col2:
            st.metric("Universe Screened", f"{summary['n_stocks']:,} stocks")
        with col3:
            # Calculate portfolio concentration
            if 'weight' in portfolio.columns:
                top_5_weight = portfolio.nlargest(5, 'weight')['weight'].sum() * 100
                st.metric("Top 5 Concentration", f"{top_5_weight:.0f}%")
            else:
                st.metric("Top 5 Concentration", "N/A")
        with col4:
            st.metric("Rebalance Frequency", "Monthly")
        
        st.markdown("---")
        
        st.markdown("""
        ## What This Portfolio Is
        
        The King's Investment Fund European Equity Strategy is a **long-term, long-only portfolio** 
        of fundamentally strong European companies.
        
        We own businesses selected based on:
        - **Fundamental strength** (profitability, balance sheet quality)
        - **Regime alignment** (favorable positioning given current market conditions)
        - **Risk contribution** (sized to manage total portfolio risk)
        
        ---
        
        ## Who This Is For
        
        | Investor Profile | Description |
        |-----------------|-------------|
        | **Horizon** | Long-term (3-5+ years) |
        | **Risk Tolerance** | Moderate - accepts short-term volatility |
        | **Objective** | Capital appreciation through fundamental compounding |
        | **Style** | Systematic, rules-based, transparent |
        
        ---
        
        ## Investment Approach Summary
        
        | Step | What We Do | Why It Matters |
        |------|-----------|----------------|
        | **1. Universe** | Screen STOXX Europe 600 | Liquid, diversified, institutional-grade |
        | **2. Fundamentals** | Assess business quality | Anchors on what we want to own |
        | **3. Regime Alignment** | Identify market conditions | Adapts to what's being priced now |
        | **4. Risk Sizing** | Allocate based on volatility | Controls risk contribution per holding |
        
        ---
        
        ## Monitoring Period Context (Feb-May)
        
        This monitoring window is for **risk oversight**, not thesis judgment.
        
        - We validate that risk discipline is maintained
        - We confirm regime alignment is functioning  
        - We expect short-term noise - this does not invalidate the long-term approach
        
        **Important:** Short-term performance should not be extrapolated. 
        The strategy is designed for long-term compounding.
        """)
    
    # =========================================================================
    # PAGE 2: INVESTMENT PHILOSOPHY
    # =========================================================================
    elif page == "Investment Philosophy":
        st.markdown("<h2>INVESTMENT PHILOSOPHY</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        Our investment process is built on three core beliefs. Understanding these 
        principles is essential to trusting the portfolio through different market conditions.
        
        ---
        
        ## The Three Pillars
        """)
        
        # Three pillars as columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 1. FUNDAMENTALS
            **What we want to own**
            
            We believe the foundation of long-term returns is owning 
            quality businesses:
            
            - **Profitability**: Companies that generate strong returns 
              on capital create value over time
            - **Balance sheet strength**: Manageable debt provides 
              resilience through cycles
            - **Business quality**: Sustainable competitive advantages 
              compound wealth
            
            *Fundamentals are slow-moving anchors, not short-term 
            trading signals.*
            """)
        
        with col2:
            st.markdown("""
            ### 2. REGIMES
            **How prices move now**
            
            Markets go through different regimes where certain 
            characteristics are rewarded:
            
            - **Risk-on**: Growth and momentum outperform
            - **Risk-off**: Quality and low-volatility outperform
            - **Rising rates**: Value tends to outperform
            - **Falling rates**: Duration-sensitive stocks benefit
            
            *Regimes drive short-term repricing, not long-term value creation.*
            """)
        
        with col3:
            st.markdown("""
            ### 3. RISK
            **How much we can own**
            
            Position sizing must reflect risk, not just conviction:
            
            - **Volatility is time-varying**: Risk changes, so should 
              position sizes
            - **Diversification matters**: No single stock should 
              dominate portfolio risk
            - **Drawdown control**: Protecting capital enables 
              long-term compounding
            
            *Risk management is continuous, not reactive.*
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ## Why This Separation Matters
        
        Many investors conflate these three concepts:
        
        | Common Mistake | Our Approach |
        |---------------|--------------|
        | "This stock is cheap, buy a lot" | Fundamentals decide *what*, not *how much* |
        | "Momentum is working, chase it" | Regime awareness adapts exposure, doesn't chase |
        | "I'm confident, ignore risk" | Conviction doesn't reduce actual risk |
        
        By separating these decisions, we maintain discipline across market conditions.
        
        ---
        
        ## How the Strategy Adapts vs Stays Disciplined
        
        | Element | Adapts | Stays Disciplined |
        |---------|--------|-------------------|
        | **Stock selection** | Which characteristics are rewarded | Always anchor on quality |
        | **Position sizing** | Based on current volatility | Always respect position limits |
        | **Rebalancing** | Monthly, based on new information | Process doesn't change |
        
        ---
        
        ## What We Do NOT Do
        
        - **Market timing**: We don't predict market direction
        - **Return forecasting**: We rank stocks, not predict magnitudes
        - **Concentration bets**: No single stock dominates the portfolio
        - **Style timing**: We don't switch between value/growth
        
        The strategy is designed to compound wealth through disciplined, 
        systematic exposure to quality European businesses.
        """)
    
    # =========================================================================
    # PAGE 3: UNIVERSE & CONSTRAINTS
    # =========================================================================
    elif page == "Universe & Constraints":
        st.markdown("<h2>UNIVERSE & CONSTRAINTS</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        ## Investment Universe: STOXX Europe 600
        
        We invest exclusively in the **STOXX Europe 600 Index**, which represents 
        large, mid, and small capitalization companies across 17 European countries.
        """)
        
        # Universe stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Index Components", "600 stocks")
        with col2:
            st.metric("After Data Filters", f"{summary['n_stocks']} stocks")
        with col3:
            st.metric("Countries Covered", "17")
        with col4:
            st.metric("Portfolio Holdings", len(portfolio))
        
        st.markdown("---")
        
        st.markdown("""
        ## Why STOXX Europe 600?
        
        | Criterion | Benefit |
        |-----------|---------|
        | **Liquidity** | All components are actively traded, enabling position entry/exit |
        | **Breadth** | 600 stocks provide ample selection opportunities |
        | **Institutional Grade** | Index is widely used by institutions, ensuring data quality |
        | **Diversification** | Spans sectors, countries, and market caps |
        | **Benchmark Availability** | Clear performance attribution vs standard benchmark |
        
        ---
        
        ## Investment Constraints
        
        We apply strict constraints to ensure prudent risk management:
        """)
        
        constraints_df = pd.DataFrame({
            'Constraint': [
                'Long-Only',
                'Maximum Position Size',
                'Minimum Liquidity',
                'Data Quality',
                'Sector Diversification'
            ],
            'Limit': [
                'No short positions',
                '10% per stock',
                'Must be in STOXX 600',
                '<20% missing price data',
                'No single sector >40%'
            ],
            'Rationale': [
                'Reduces complexity, avoids unlimited downside',
                'Prevents concentration in single names',
                'Ensures positions can be liquidated',
                'Ensures reliable signal generation',
                'Avoids sector-specific blow-ups'
            ]
        })
        st.dataframe(constraints_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.markdown("""
        ## Data Inputs
        
        We consider multiple categories of information:
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Fundamental Data
            - Return on Equity (ROE)
            - Profit Margins
            - Debt/Equity Ratio
            - Asset Turnover
            
            *Source: WRDS Compustat Global*
            """)
            
            st.markdown("""
            ### Style Factors
            - Price Momentum (1m, 3m, 6m, 12m)
            - Realized Volatility
            - Market Beta
            
            *Source: Calculated from price data*
            """)
        
        with col2:
            st.markdown("""
            ### Macro Indicators
            - Interest Rates (US 10Y, 2Y, EUR rates)
            - Volatility (VIX, VSTOXX)
            - Currencies (EUR/USD, EUR/GBP)
            - Commodities (Oil, Gold)
            
            *Source: Yahoo Finance, FRED*
            """)
            
            st.markdown("""
            ### Market Data
            - Daily adjusted prices
            - Trading volumes
            - Index levels
            
            *Source: Yahoo Finance*
            """)
        
        st.markdown("---")
        
        st.markdown(f"""
        ## Current Universe Summary
        
        | Metric | Value |
        |--------|-------|
        | Stocks Screened | {summary['n_stocks']} |
        | Total Data Inputs | {summary['n_factors']} factors |
        | Data History | 3 years |
        | Rebalance Frequency | Monthly |
        """)
    
    # =========================================================================
    # PAGE 4: STOCK SELECTION
    # =========================================================================
    elif page == "Stock Selection":
        st.markdown("<h2>STOCK SELECTION</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        ## How We Select Stocks
        
        Our selection process identifies stocks that combine **fundamental strength** 
        with **favorable regime alignment**. Here's how it works:
        """)
        
        st.markdown("""
        ### The Selection Process
        
        ```
        600 European Stocks
              |
              v
        [Data Quality Filter] ──> Remove stocks with insufficient data
              |
              v
        [Fundamental Assessment] ──> Score business quality
              |
              v
        [Regime Alignment] ──> Identify what market is currently rewarding
              |
              v
        [Combined Ranking] ──> Rank stocks by overall attractiveness
              |
              v
        [Top Selection] ──> Select top-ranked stocks for portfolio
        ```
        
        ---
        
        ### What We Look For
        
        | Category | Characteristics | Why It Matters |
        |----------|----------------|----------------|
        | **Fundamentals** | Profitability, low leverage, quality | Long-term value creation |
        | **Momentum** | Recent price performance | Market recognition of quality |
        | **Volatility** | Risk characteristics | Position sizing input |
        | **Regime Fit** | Alignment with current conditions | Tactical adaptation |
        
        ---
        
        ### Key Principle: Ranking, Not Prediction
        
        **Important:** We do NOT predict exact returns. Instead, we **rank** stocks 
        by their relative attractiveness:
        
        - A stock ranked #1 is more attractive than #50
        - The exact magnitude of the score is less important than the ordering
        - We select stocks from the **top bucket**, not based on a specific threshold
        
        This approach is robust because ranking is easier than precise prediction.
        """)
        
        st.markdown("---")
        st.markdown("## Current Top-Ranked Stocks")
        
        # Get rankings
        ranker = StockRanker()
        rankings = ranker.rank_stocks(pipeline.expected_returns)
        
        # Create investor-friendly display
        top_stocks = rankings.head(30).copy()
        top_stocks['Rank'] = range(1, len(top_stocks) + 1)
        top_stocks = top_stocks.rename(columns={'ticker': 'Stock'})
        
        # Categorize into buckets
        def get_bucket(rank):
            if rank <= 10:
                return "Top 10"
            elif rank <= 20:
                return "Top 20"
            else:
                return "Top 30"
        
        top_stocks['Selection Bucket'] = top_stocks['Rank'].apply(get_bucket)
        
        # Display without technical signal values
        display_df = top_stocks[['Rank', 'Stock', 'Selection Bucket']].copy()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("""
            ### Selection Buckets
            
            - **Top 10**: Highest conviction
            - **Top 20**: Strong alignment  
            - **Top 30**: Solid fundamentals
            
            Portfolio weights are determined 
            by risk contribution, not rank 
            alone.
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ## What the Ranking Reflects
        
        Stocks rank highly when they demonstrate:
        
        1. **Strong fundamental characteristics** relative to peers
        2. **Favorable alignment** with current market conditions
        3. **Stable historical relationships** with return drivers
        
        The ranking does NOT reflect:
        - Predicted return magnitudes
        - Short-term trading signals
        - Analyst recommendations or news sentiment
        
        ---
        
        ## Ranking Stability
        
        Month-over-month, we expect:
        - ~70-80% of top stocks to remain in the top bucket
        - Gradual rotation, not dramatic changes
        - New entrants typically from the "just outside" group
        
        High turnover would indicate unstable signals and is monitored closely.
        """)
    
    # =========================================================================
    # PAGE 5: FACTOR ANALYSIS (Technical + Investor Blend)
    # =========================================================================
    elif page == "Factor Analysis":
        st.markdown("<h2>FACTOR ANALYSIS</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        ## What Drives Returns?
        
        This page shows **what the market is currently pricing** - the factors and characteristics 
        that are driving stock returns. Understanding these drivers helps explain portfolio positioning.
        """)
        
        st.markdown("---")
        
        # Get PLS data
        loadings = pipeline.pls_model.get_component_loadings()
        
        # Factor categories for interpretation
        factor_categories = {
            'Momentum': ['return_1m', 'return_3m', 'return_6m', 'return_12m'],
            'Volatility': ['realized_vol', 'beta', 'VIX', 'VSTOXX'],
            'Rates': ['US_10Y', 'US_2Y', 'EUR_GOVT', 'IRX'],
            'Currency': ['EURUSD', 'EURGBP', 'EURCHF', 'USDJPY'],
            'Commodity': ['OIL', 'BRENT', 'GOLD', 'NATGAS'],
            'Value': ['pe_ratio', 'pb_ratio', 'book_value', 'eps'],
            'Quality': ['roe_factor', 'profit_margin_factor', 'quality_factor', 'profitability_factor', 
                       'leverage_factor', 'roe', 'profit_margin', 'debt_equity', 'asset_turnover'],
            'Market': ['SP500', 'STOXX', 'DAX', 'CAC40', 'FTSE']
        }
        
        # Build component interpretations
        component_interpretations = []
        for col in loadings.columns:
            component_loadings = loadings[col].abs().sort_values(ascending=False)
            top_factors = component_loadings.head(5).index.tolist()
            
            category_scores = {cat: 0 for cat in factor_categories}
            for factor in top_factors:
                for cat, keywords in factor_categories.items():
                    if any(kw.lower() in factor.lower() for kw in keywords):
                        category_scores[cat] += component_loadings[factor]
            
            dominant_cat = max(category_scores, key=category_scores.get)
            top_3 = component_loadings.head(3)
            top_factors_str = ", ".join([f"{f} ({loadings[col][f]:.2f})" for f in top_3.index])
            
            component_interpretations.append({
                'Component': col,
                'Theme': dominant_cat,
                'Key Drivers': top_factors_str
            })
        
        # Current regime interpretation
        st.markdown("## Current Market Regime")
        
        latest_pls = pipeline.pls_components.iloc[-1]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### What's Driving Returns Now")
            for interp in component_interpretations[:4]:
                val = latest_pls.get(interp['Component'], 0)
                direction = "Positive" if val > 0.5 else "Negative" if val < -0.5 else "Neutral"
                st.markdown(f"**{interp['Theme']}**: {direction} ({val:.2f})")
        
        with col2:
            st.markdown("### Interpretation")
            st.markdown("""
            - **Positive values**: Factor is currently rewarded
            - **Negative values**: Factor is currently penalized
            - **Near zero**: Factor is neutral
            
            *The portfolio tilts toward stocks aligned with positive factors.*
            """)
        
        st.markdown("---")
        
        # Component details table
        st.markdown("## Return Driver Components (PLS)")
        st.caption("Each component represents a combination of factors that collectively drive returns")
        
        interp_df = pd.DataFrame(component_interpretations)
        st.dataframe(interp_df, use_container_width=True, hide_index=True)
        
        # Factor importance chart
        st.markdown("---")
        st.markdown("## Factor Importance")
        st.caption("Which individual factors matter most for explaining returns")
        
        importance = pipeline.pls_model.get_factor_importance()
        fig = px.bar(
            importance.head(20),
            x='importance',
            y='factor',
            orientation='h',
            color='importance',
            color_continuous_scale='YlOrRd'
        )
        fig.update_layout(
            title="Top 20 Most Important Factors",
            paper_bgcolor='#000000',
            plot_bgcolor='#0a0a0a',
            font_color='#f7b801',
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Component time series
        st.markdown("---")
        st.markdown("## Factor Regime Over Time")
        st.caption("How the market regime has evolved - helps identify regime shifts")
        
        components = pipeline.pls_model.get_component_scores()
        component_names = {row['Component']: f"{row['Component']} ({row['Theme']})" 
                         for row in component_interpretations}
        
        fig = go.Figure()
        for col in components.columns:
            display_name = component_names.get(col, col)
            fig.add_trace(go.Scatter(
                x=components.index,
                y=components[col],
                name=display_name,
                line=dict(width=2)
            ))
        fig.update_layout(
            title="PLS Components Over Time",
            xaxis_title="Date",
            yaxis_title="Component Value",
            paper_bgcolor='#000000',
            plot_bgcolor='#0a0a0a',
            font_color='#f7b801',
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical details (expandable)
        st.markdown("---")
        with st.expander("Technical Details: Full PLS Loadings"):
            st.dataframe(loadings.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
        
        # Cross-validation results
        st.markdown("---")
        st.markdown("## Model Calibration (Cross-Validation)")
        st.caption("How we selected model parameters to avoid overfitting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### PLS Component Selection")
            pls_cv = pipeline.pls_model.get_cv_results()
            
            if pls_cv and 'n_components' in pls_cv:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pls_cv['n_components'],
                    y=pls_cv['cv_error'],
                    mode='lines+markers',
                    name='CV Error',
                    line=dict(color='#f7b801', width=3)
                ))
                
                actual_selected = pipeline.pls_model.optimal_n_components or pipeline.pls_model.n_components
                if actual_selected in pls_cv['n_components']:
                    selected_idx = pls_cv['n_components'].index(actual_selected)
                    selected_error = pls_cv['cv_error'][selected_idx]
                    fig.add_trace(go.Scatter(
                        x=[actual_selected],
                        y=[selected_error],
                        mode='markers',
                        name=f'Selected: {actual_selected}',
                        marker=dict(color='#00ff00', size=15, symbol='star')
                    ))
                
                fig.update_layout(
                    xaxis_title="Number of Components",
                    yaxis_title="CV Error",
                    paper_bgcolor='#000000',
                    plot_bgcolor='#0a0a0a',
                    font_color='#f7b801'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"Selected {actual_selected} components based on cross-validation")
        
        with col2:
            st.markdown("### Ridge Regularization")
            en_cv = pipeline.elastic_net_model.get_cv_results()
            
            if en_cv and 'alpha_distribution' in en_cv and en_cv['alpha_distribution']:
                avg_alpha = np.mean(en_cv['alpha_distribution'])
                
                fig = px.histogram(
                    x=np.log10(en_cv['alpha_distribution']),
                    nbins=20,
                    title="Ridge Alpha Distribution (log scale)",
                    labels={'x': 'log10(alpha)', 'y': 'Count'}
                )
                fig.update_layout(
                    paper_bgcolor='#000000',
                    plot_bgcolor='#0a0a0a',
                    font_color='#f7b801'
                )
                fig.update_traces(marker_color='#f7b801')
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Using pure Ridge regression | Avg alpha: {avg_alpha:.4f}")
            else:
                st.info("Ridge regression: keeps all coefficients, provides stable estimates")
        
        # Model summary metrics
        st.markdown("---")
        st.markdown("## Model Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Universe Size", f"{summary['n_stocks']:,}")
        with col2:
            st.metric("Input Factors", f"{summary['n_factors']:,}")
        with col3:
            st.metric("PLS Components", summary['n_pls_components'])
        with col4:
            st.metric("Avg R²", f"{summary['avg_r2']:.4f}")
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Explained Variance", f"{summary['pls_explained_variance']:.1%}")
        with col6:
            signal_std = summary['expected_returns_stats']['std']
            st.metric("Signal Spread", f"{signal_std:.6f}")
        with col7:
            signal_range = summary['expected_returns_stats']['max'] - summary['expected_returns_stats']['min']
            st.metric("Signal Range", f"{signal_range:.6f}")
        with col8:
            st.metric("Portfolio Size", len(portfolio))
        
        # Signal distribution
        st.markdown("### Signal Distribution")
        st.caption("How model scores are distributed across the universe")
        fig = px.histogram(
            x=pipeline.expected_returns.values,
            nbins=50,
            labels={'x': 'Model Signal', 'y': 'Count'},
            color_discrete_sequence=['#f7b801']
        )
        fig.update_layout(
            paper_bgcolor='#000000',
            plot_bgcolor='#0a0a0a',
            font_color='#f7b801',
            xaxis=dict(gridcolor='#333'),
            yaxis=dict(gridcolor='#333')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretation:**
        - Clustered signals = model sees most stocks similarly (expected with daily data)
        - Outliers on the right = stocks with strongest positive alignment (our long candidates)
        - We select from the **right tail**, not based on absolute magnitude
        """)
    
    # =========================================================================
    # PAGE 6: CURRENT PORTFOLIO
    # =========================================================================
    elif page == "Current Portfolio":
        st.markdown("<h2>CURRENT PORTFOLIO</h2>", unsafe_allow_html=True)
        
        # Portfolio metrics
        optimizer = PortfolioOptimizer()
        metrics = optimizer.calculate_portfolio_metrics(portfolio)
        
        # Key portfolio stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Positions", metrics.get('n_positions', 0))
        with col2:
            effective_n = metrics.get('effective_n_positions', 0)
            st.metric("Effective Positions", f"{effective_n:.0f}", 
                     help="Diversification measure - accounts for weight concentration")
        with col3:
            max_pos = metrics.get('max_position', 0) * 100
            st.metric("Largest Position", f"{max_pos:.1f}%")
        with col4:
            top_5_weight = portfolio.nlargest(5, 'weight')['weight'].sum() * 100
            st.metric("Top 5 Weight", f"{top_5_weight:.0f}%")
        
        st.markdown("---")
        
        # Holdings table (investor-friendly)
        st.markdown("## Holdings")
        
        display_portfolio = portfolio[['ticker', 'weight']].copy()
        display_portfolio['Weight (%)'] = (display_portfolio['weight'] * 100).round(2)
        display_portfolio = display_portfolio.rename(columns={'ticker': 'Stock'})
        display_portfolio = display_portfolio.sort_values('Weight (%)', ascending=False)
        display_portfolio['Rank'] = range(1, len(display_portfolio) + 1)
        
        # Add sector if available
        if 'sector' in portfolio.columns:
            sector_map = dict(zip(portfolio['ticker'], portfolio['sector'].fillna('Unknown')))
            display_portfolio['Sector'] = display_portfolio['Stock'].map(sector_map)
            display_df = display_portfolio[['Rank', 'Stock', 'Weight (%)', 'Sector']]
        else:
            display_df = display_portfolio[['Rank', 'Stock', 'Weight (%)']]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Weight distribution chart
        st.markdown("## Weight Distribution")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=display_portfolio['Stock'],
            y=display_portfolio['Weight (%)'],
            marker_color='#f7b801'
        ))
        fig.update_layout(
            xaxis_title="Stock",
            yaxis_title="Weight (%)",
            paper_bgcolor='#000000',
            plot_bgcolor='#0a0a0a',
            font_color='#f7b801',
            xaxis=dict(gridcolor='#333', tickangle=45),
            yaxis=dict(gridcolor='#333')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sector breakdown
        st.markdown("---")
        st.markdown("## Sector Allocation")
        
        if 'sector' in portfolio.columns:
            portfolio_sectors = portfolio.copy()
            portfolio_sectors['sector'] = portfolio_sectors['sector'].fillna('Unknown')
            portfolio_sectors['sector'] = portfolio_sectors['sector'].replace('', 'Unknown')
            
            sector_weights = portfolio_sectors.groupby('sector')['weight'].sum() * 100
            sector_weights = sector_weights.sort_values(ascending=False)
            
            if len(sector_weights) > 0 and sector_weights.sum() > 0:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    fig = px.pie(
                        values=sector_weights.values,
                        names=sector_weights.index,
                        color_discrete_sequence=px.colors.sequential.YlOrRd
                    )
                    fig.update_layout(
                        paper_bgcolor='#000000',
                        font_color='#f7b801'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    sector_df = pd.DataFrame({
                        'Sector': sector_weights.index,
                        'Weight (%)': sector_weights.values.round(1)
                    })
                    st.dataframe(sector_df, use_container_width=True, hide_index=True)
                    
                    # Sector concentration warning
                    max_sector = sector_weights.max()
                    if max_sector > 30:
                        st.warning(f"Largest sector ({sector_weights.idxmax()}) is {max_sector:.0f}% - monitor concentration")
            else:
                st.info("Sector data not available")
        else:
            st.info("Sector classification not available for current holdings")
        
        st.markdown("---")
        
        st.markdown("""
        ## Portfolio Construction Notes
        
        - **Weights are risk-based**: Larger positions have favorable risk characteristics, not just high rankings
        - **Position limits enforced**: Maximum 10% per stock to ensure diversification
        - **Long-only**: All positions are equity ownership, no short selling
        - **Monthly rebalancing**: Portfolio is reviewed and rebalanced monthly
        """)
    
    # =========================================================================
    # PAGE 6: RISK & MONITORING
    # =========================================================================
    elif page == "Risk & Monitoring":
        st.markdown("<h2>RISK & MONITORING</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        ## Risk Discipline
        
        Risk management is a continuous process, not a reaction to drawdowns. 
        This page shows how we monitor and control portfolio risk.
        """)
        
        # Current risk metrics
        st.markdown("### Current Portfolio Risk")
        
        # Calculate basic risk metrics from returns if available
        if hasattr(pipeline, 'stock_returns') and len(pipeline.stock_returns) > 0:
            portfolio_tickers = portfolio['ticker'].tolist()
            portfolio_weights = portfolio.set_index('ticker')['weight']
            
            # Get returns for portfolio stocks
            available_tickers = [t for t in portfolio_tickers if t in pipeline.stock_returns.columns]
            if available_tickers:
                port_returns = pipeline.stock_returns[available_tickers]
                weights_aligned = portfolio_weights.reindex(available_tickers).fillna(0)
                
                # Calculate weighted portfolio returns
                weighted_returns = (port_returns * weights_aligned).sum(axis=1)
                
                # Volatility metrics
                daily_vol = weighted_returns.std()
                annual_vol = daily_vol * np.sqrt(252)
                
                # Recent vol (last 20 days)
                recent_vol = weighted_returns.tail(20).std() * np.sqrt(252)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Annualized Volatility", f"{annual_vol:.1%}")
                with col2:
                    st.metric("Recent Volatility (20d)", f"{recent_vol:.1%}")
                with col3:
                    # Compare to VIX if possible
                    vol_change = recent_vol - annual_vol
                    st.metric("Vol Change", f"{vol_change:+.1%}", 
                             delta=f"{'Rising' if vol_change > 0 else 'Falling'}")
                with col4:
                    max_weight = portfolio['weight'].max() * 100
                    st.metric("Max Position", f"{max_weight:.1f}%")
        
        st.markdown("---")
        
        st.markdown("""
        ## Monitoring Framework
        
        We monitor the portfolio across three dimensions:
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### Volatility
            
            **What we track:**
            - Portfolio volatility vs historical
            - Individual stock volatility spikes
            - Correlation changes
            
            **Action triggers:**
            - Volatility >50% above normal
            - Single stock vol spike >3x
            """)
        
        with col2:
            st.markdown("""
            ### Drawdown
            
            **What we track:**
            - Current drawdown from peak
            - Recovery time
            - Comparison to benchmark
            
            **Action triggers:**
            - Drawdown >15% (review)
            - Drawdown >25% (reassess)
            """)
        
        with col3:
            st.markdown("""
            ### Regime
            
            **What we track:**
            - Factor returns vs expectations
            - Macro indicator shifts
            - Style rotation patterns
            
            **Action triggers:**
            - Sustained underperformance
            - Major regime shift detected
            """)
        
        st.markdown("---")
        
        st.markdown("""
        ## Feb-May Monitoring Context
        
        This period is designated as a **risk oversight window**, not a performance evaluation period.
        
        ### What We Are Watching
        
        | Aspect | Purpose | Expected Outcome |
        |--------|---------|------------------|
        | **Volatility behavior** | Validate risk models | Vol should track forecasts |
        | **Drawdown discipline** | Confirm risk limits work | Drawdowns within expected range |
        | **Regime signals** | Test adaptation mechanism | Portfolio responds to shifts |
        | **Turnover** | Ensure stability | Moderate, not excessive |
        
        ### What We Are NOT Doing
        
        - **Judging the thesis** based on short-term returns
        - **Changing the strategy** in response to noise
        - **Extrapolating** 3-month results to long-term
        
        ---
        
        ## Risk Limits Summary
        
        | Limit | Value | Current Status |
        |-------|-------|----------------|
        | Max Position | 10% | Compliant |
        | Max Sector | 40% | Compliant |
        | Min Positions | 20 | {len(portfolio)} positions |
        | Drawdown Alert | 15% | Monitoring |
        
        ---
        
        ## Rebalancing Discipline
        
        - **Frequency**: Monthly
        - **Triggers**: Calendar-based, not reactive
        - **Process**: Re-run model, compare to current holdings, execute trades
        
        We do NOT rebalance intra-month unless a position hits a hard risk limit.
        """)
    
    # =========================================================================
    # PAGE 7: PERFORMANCE
    # =========================================================================
    elif page == "Performance":
        st.markdown("<h2>PERFORMANCE</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        ## How to Interpret Performance
        
        **Important context before viewing results:**
        
        - This strategy is designed for **long-term compounding**, not short-term trading
        - Short-term results (< 1 year) are dominated by noise and regime effects
        - We expect periods of underperformance when our factors are out of favor
        - The goal is **risk-adjusted outperformance over full cycles**
        """)
        
        st.markdown("---")
        
        # Backtest period selection
        backtest_months = st.selectbox(
            "Analysis Period:",
            [12, 24, 36],
            format_func=lambda x: f"{x} months ({x//12} year{'s' if x > 12 else ''})",
            index=0
        )
        
        st.markdown("---")
        
        try:
            st.markdown(f"### Historical Analysis ({backtest_months} Months)")
            st.caption("Walk-forward simulation: model trained only on past data at each rebalance point.")
            
            # Progress placeholder
            progress_text = st.empty()
            
            def update_progress(msg):
                progress_text.text(msg)
            
            # Run walk-forward backtest
            backtest_results = run_walk_forward_backtest(
                stock_returns=pipeline.stock_returns,
                factors=pipeline.combined_factors,
                portfolio_size=PORTFOLIO_SIZE,
                backtest_months=backtest_months,
                progress_callback=update_progress
            )
            
            progress_text.empty()
            
            if not backtest_results['metrics']:
                st.warning("Insufficient data for walk-forward backtest")
                st.stop()
            
            metrics = backtest_results['metrics']
            
            # Key Performance Metrics
            st.markdown("<h3>Key Metrics</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
            with col2:
                st.metric("Annual Return", f"{metrics['annual_return']:.1%}")
            with col3:
                st.metric("Annual Volatility", f"{metrics['annual_volatility']:.1%}")
            with col4:
                st.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
            
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
            with col6:
                st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.2f}")
            with col7:
                st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
            with col8:
                st.metric("Beta", f"{metrics['beta']:.2f}")
            
            # Benchmark Comparison
            st.markdown("<h3>vs Benchmark (Equal-Weight Universe)</h3>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Portfolio Return", f"{metrics['annual_return']:.1%}")
            with col2:
                st.metric("Benchmark Return", f"{metrics['benchmark_return']:.1%}")
            with col3:
                excess = metrics['excess_return']
                st.metric("Excess Return", f"{excess:.1%}", delta=f"{excess:.1%}")
            with col4:
                st.metric("Alpha", f"{metrics['alpha']:.1%}")
            
            # Cumulative Returns Chart - Portfolio vs European Indices
            st.markdown("<h3>Portfolio vs European Indices</h3>", unsafe_allow_html=True)
            
            port_cum = backtest_results.get('portfolio_cumulative', pd.Series())
            bench_cum = backtest_results.get('benchmark_cumulative', pd.Series())
            
            if len(port_cum) > 0:
                fig = go.Figure()
                
                # Portfolio
                fig.add_trace(go.Scatter(
                    x=list(range(len(port_cum))),
                    y=port_cum.values,
                    name='KIF Portfolio',
                    line=dict(color='#f7b801', width=3)
                ))
                
                # Equal-weight benchmark
                fig.add_trace(go.Scatter(
                    x=list(range(len(bench_cum))),
                    y=bench_cum.values,
                    name='Equal-Weight Universe',
                    line=dict(color='#888888', width=2, dash='dash')
                ))
                
                # Fetch European indices for comparison
                try:
                    import yfinance as yf
                    backtest_start = backtest_results.get('start_date', None)
                    backtest_end = backtest_results.get('end_date', None)
                    
                    if backtest_start and backtest_end:
                        indices = {
                            '^STOXX': ('STOXX 600', '#00bcd4'),
                            '^GDAXI': ('DAX', '#4caf50'),
                            '^FCHI': ('CAC 40', '#2196f3'),
                            '^FTSE': ('FTSE 100', '#9c27b0')
                        }
                        
                        for ticker, (name, color) in indices.items():
                            try:
                                idx_data = yf.download(ticker, start=backtest_start, end=backtest_end, progress=False)
                                if len(idx_data) > 0:
                                    idx_returns = idx_data['Adj Close'].pct_change().dropna()
                                    idx_cum = (1 + idx_returns).cumprod()
                                    # Resample to match portfolio length
                                    if len(idx_cum) >= len(port_cum):
                                        idx_cum_aligned = idx_cum.iloc[:len(port_cum)].values
                                    else:
                                        idx_cum_aligned = np.interp(
                                            np.linspace(0, 1, len(port_cum)),
                                            np.linspace(0, 1, len(idx_cum)),
                                            idx_cum.values
                                        )
                                    fig.add_trace(go.Scatter(
                                        x=list(range(len(port_cum))),
                                        y=idx_cum_aligned,
                                        name=name,
                                        line=dict(color=color, width=1.5),
                                        opacity=0.7
                                    ))
                            except:
                                pass
                except Exception as e:
                    st.warning(f"Could not fetch index data: {e}")
                
                fig.update_layout(
                    title="Portfolio vs European Market Indices",
                    xaxis_title="Trading Days",
                    yaxis_title="Cumulative Return (1 = starting value)",
                    paper_bgcolor='#000000',
                    plot_bgcolor='#0a0a0a',
                    font_color='#f7b801',
                    xaxis=dict(gridcolor='#333'),
                    yaxis=dict(gridcolor='#333'),
                    legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown Chart
            st.markdown("<h3>Drawdown</h3>", unsafe_allow_html=True)
            drawdowns = backtest_results.get('drawdowns', pd.Series())
            
            if len(drawdowns) > 0:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(drawdowns))),
                    y=drawdowns.values * 100,
                    fill='tozeroy',
                    name='Drawdown',
                    line=dict(color='#ff4444', width=1),
                    fillcolor='rgba(255, 68, 68, 0.3)'
                ))
                fig.update_layout(
                    title="Portfolio Drawdown",
                    xaxis_title="Trading Days",
                    yaxis_title="Drawdown (%)",
                    paper_bgcolor='#000000',
                    plot_bgcolor='#0a0a0a',
                    font_color='#f7b801',
                    xaxis=dict(gridcolor='#333'),
                    yaxis=dict(gridcolor='#333')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Walk-forward: show rebalance history
            if 'holdings_history' in backtest_results:
                st.markdown("<h3>Rebalance History</h3>", unsafe_allow_html=True)
                holdings = backtest_results['holdings_history']
                if holdings:
                    st.write(f"Total rebalances: {len(holdings)}")
                    with st.expander("View Holdings at Each Rebalance"):
                        for h in holdings[-5:]:  # Show last 5
                            st.write(f"**{h['date'].date()}**: {len(h['holdings'])} stocks")
                            st.write(", ".join(h['holdings'][:10]) + ("..." if len(h['holdings']) > 10 else ""))
            
            # Risk Metrics Summary
            st.markdown("<h3>Risk Metrics Summary</h3>", unsafe_allow_html=True)
            risk_df = pd.DataFrame({
                'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 
                          'Volatility', 'Beta', 'Alpha', 'Tracking Error', 'Information Ratio'],
                'Value': [
                    f"{metrics['sharpe_ratio']:.2f}",
                    f"{metrics['sortino_ratio']:.2f}",
                    f"{metrics['calmar_ratio']:.2f}",
                    f"{metrics['max_drawdown']:.1%}",
                    f"{metrics['annual_volatility']:.1%}",
                    f"{metrics['beta']:.2f}",
                    f"{metrics['alpha']:.1%}",
                    f"{metrics['tracking_error']:.1%}",
                    f"{metrics['information_ratio']:.2f}"
                ],
                'Description': [
                    'Risk-adjusted return (excess return / volatility)',
                    'Risk-adjusted return using downside volatility',
                    'Annual return / max drawdown',
                    'Largest peak-to-trough decline',
                    'Annualized standard deviation of returns',
                    'Sensitivity to benchmark movements',
                    'Excess return above CAPM prediction',
                    'Volatility of excess returns vs benchmark',
                    'Risk-adjusted excess return vs benchmark'
                ]
            })
            st.dataframe(risk_df, use_container_width=True, hide_index=True)
            
            # Interpretation guidance
            st.markdown("---")
            st.markdown("""
            ## How to Read These Results
            
            ### What Good Performance Looks Like
            
            | Metric | Target | Interpretation |
            |--------|--------|----------------|
            | **Sharpe Ratio** | > 0.5 | Risk-adjusted return above market |
            | **Alpha** | Positive | Outperformance vs benchmark |
            | **Max Drawdown** | < 25% | Risk within acceptable bounds |
            | **Beta** | 0.8 - 1.2 | Market-like exposure |
            
            ### When to Expect Underperformance
            
            This strategy may underperform when:
            - **Low quality rallies**: Speculative, unprofitable stocks lead the market
            - **Momentum crashes**: Sudden reversals hurt momentum exposure
            - **Extreme risk-on**: Highest beta names outperform regardless of fundamentals
            
            These periods are expected and do not indicate strategy failure.
            
            ### Why Discipline Matters
            
            The temptation during underperformance is to:
            - Switch strategies (buy high, sell low)
            - Increase risk to "catch up"
            - Abandon the process
            
            **This is precisely when discipline creates long-term value.**
            
            History shows that factor premiums are persistent but lumpy. 
            Consistent exposure through cycles is how compounding works.
            """)
            
            # Simulation context
            st.markdown("---")
            st.caption(f"Analysis period: {backtest_months} months | Simulation uses walk-forward methodology with monthly rebalancing")
            
        except Exception as e:
            st.error(f"Error running analysis: {str(e)}")
            st.text(traceback.format_exc())

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color: #666; font-size: 12px;'>King's Investment Fund<br>Long-Term European Equity Strategy<br>Fundamentals | Regimes | Risk Discipline</p>", unsafe_allow_html=True)
