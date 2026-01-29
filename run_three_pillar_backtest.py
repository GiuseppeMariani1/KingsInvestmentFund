"""
THREE-PILLAR BACKTEST - FULL TRANSPARENCY
=========================================
This script runs the three-pillar backtest with complete visibility
into what each pillar is actually doing.
"""
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from investment_dashboard.data.hybrid_loader import HybridEquityUniverse
from investment_dashboard.data.style_factors import StyleFactorCalculator
from investment_dashboard.data.macro_factors import MacroFactorLoader
from investment_dashboard.portfolio.three_pillar_backtest import run_three_pillar_backtest

print("=" * 80)
print("THREE-PILLAR INVESTMENT STRATEGY - FULL TRANSPARENCY RUN")
print("=" * 80)
print(f"Run at: {datetime.now()}")

# =============================================================================
# STEP 1: Load Universe
# =============================================================================
print("\n" + "=" * 80)
print("STEP 1: UNIVERSE")
print("=" * 80)

universe = HybridEquityUniverse()
universe.build_universe()

print(f"Universe: {len(universe.stocks)} stocks")

# =============================================================================
# STEP 2: Load Returns (3 years)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 2: RETURNS DATA")
print("=" * 80)

end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)

print(f"Fetching returns from {start_date.date()} to {end_date.date()}...")
returns = universe.fetch_returns(start_date, end_date)
print(f"Returns: {returns.shape[1]} stocks, {returns.shape[0]} days")

# =============================================================================
# STEP 3: Load Fundamentals (PILLAR 1)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 3: FUNDAMENTALS (Pillar 1 - What to Own)")
print("=" * 80)

# Get fundamentals from WRDS via style factor calculator
calculator = StyleFactorCalculator()
tickers = returns.columns.tolist()

print(f"Fetching fundamentals for {len(tickers)} tickers...")
fundamentals = calculator._fetch_wrds_fundamentals(tickers)

if fundamentals is not None and not fundamentals.empty:
    print(f"\nFundamentals loaded:")
    print(f"  Companies in WRDS: {len(fundamentals)}")
    
    # Create a clean fundamentals DataFrame indexed by ticker
    # We need to match WRDS data to our tickers
    fund_matched = {}
    
    for _, row in fundamentals.iterrows():
        company = str(row.get('company_name', '')).upper()
        if not company:
            continue
            
        for ticker in tickers:
            base = ticker.split('.')[0].upper() if '.' in ticker else ticker.upper()
            if len(base) >= 3 and base in company:
                fund_matched[ticker] = {
                    'roe': row.get('roe', np.nan),
                    'debt_equity': row.get('debt_equity', np.nan),
                    'profit_margin': row.get('profit_margin', np.nan),
                    'market_cap': row.get('market_cap', np.nan),
                }
                break
    
    fundamentals_df = pd.DataFrame(fund_matched).T
    fundamentals_df.index.name = 'ticker'
    
    print(f"  Matched to tickers: {len(fundamentals_df)}")
    print(f"\n  Fundamental coverage:")
    for col in ['roe', 'debt_equity', 'profit_margin']:
        if col in fundamentals_df.columns:
            valid = fundamentals_df[col].notna().sum()
            print(f"    - {col}: {valid}/{len(fundamentals_df)} ({valid/len(fundamentals_df)*100:.0f}%)")
    
    print(f"\n  Sample fundamentals (first 10):")
    print(fundamentals_df.head(10))
else:
    print("  WARNING: No fundamentals available!")
    fundamentals_df = pd.DataFrame()

# =============================================================================
# STEP 4: Load Regime Factors (PILLAR 2)
# =============================================================================
print("\n" + "=" * 80)
print("STEP 4: REGIME FACTORS (Pillar 2 - When to Own)")
print("=" * 80)

# Load macro factors
print("Loading macro factors...")
macro_loader = MacroFactorLoader()
macro_factors = macro_loader.load_all_factors(start_date, end_date)
print(f"  Macro factors: {len(macro_factors.columns)} series")

# Load style factors (ONLY time-varying ones!)
print("\nCalculating style factors (time-varying only)...")

# We need prices for style factors
import yfinance as yf
sample_tickers = tickers[:100]  # Use sample for speed in this demo

try:
    prices_raw = yf.download(sample_tickers, start=start_date, end=end_date, progress=False)
    if isinstance(prices_raw.columns, pd.MultiIndex):
        prices = prices_raw['Close'] if 'Close' in prices_raw.columns.get_level_values(0) else None
    else:
        prices = prices_raw
    
    if prices is not None:
        prices = prices.dropna(axis=1, how='all')
        sample_returns = returns[prices.columns] if len(prices.columns) > 0 else None
    else:
        sample_returns = None
except Exception as e:
    print(f"  Price download error: {e}")
    sample_returns = None

if sample_returns is not None and not sample_returns.empty:
    style_factors_raw = calculator.calculate_rolling_style_factors(prices, sample_returns)
    
    # FILTER: Only keep time-varying factors (variance > 0.001)
    time_varying = []
    for col in style_factors_raw.columns:
        if style_factors_raw[col].var() > 0.001:
            time_varying.append(col)
    
    style_factors = style_factors_raw[time_varying]
    print(f"  Style factors (time-varying): {len(style_factors.columns)}")
    print(f"  Filtered out {len(style_factors_raw.columns) - len(time_varying)} constant factors")
else:
    style_factors = pd.DataFrame()

# Combine factors for regime detection
print("\nCombining regime factors...")
if not style_factors.empty:
    factors = pd.concat([macro_factors, style_factors], axis=1)
else:
    factors = macro_factors

# Align
common_dates = returns.index.intersection(factors.index)
factors = factors.loc[common_dates]
returns_aligned = returns.loc[common_dates]

print(f"  Combined factors: {len(factors.columns)} series, {len(factors)} days")

# Show factor list
print(f"\n  Factor list (for regime detection):")
for i, col in enumerate(factors.columns[:20]):
    print(f"    {i+1}. {col}")
if len(factors.columns) > 20:
    print(f"    ... and {len(factors.columns) - 20} more")

# =============================================================================
# STEP 5: Run Three-Pillar Backtest
# =============================================================================
print("\n" + "=" * 80)
print("STEP 5: THREE-PILLAR BACKTEST")
print("=" * 80)

# Run with different quality weights to show impact
quality_weights = [0.0, 0.3, 0.5]

for qw in quality_weights:
    print(f"\n{'=' * 60}")
    print(f"QUALITY WEIGHT: {qw:.0%}")
    print(f"{'=' * 60}")
    
    results = run_three_pillar_backtest(
        stock_returns=returns_aligned,
        factors=factors,
        fundamentals=fundamentals_df,
        portfolio_size=30,
        backtest_months=36,  # 3 years
        quality_weight=qw
    )
    
    m = results['metrics']
    print(f"\n  RESULTS (Quality Weight = {qw:.0%}):")
    print(f"    Sharpe Ratio:    {m['sharpe_ratio']:.2f}")
    print(f"    Annual Return:   {m['annual_return']:.1%}")
    print(f"    Annual Vol:      {m['annual_volatility']:.1%}")
    print(f"    Max Drawdown:    {m['max_drawdown']:.1%}")
    print(f"    Benchmark:       {m['benchmark_return']:.1%}")
    print(f"    Alpha:           {m['alpha']:.1%}")
    
    # Quality diagnostics
    if not results.get('quality_scores', pd.DataFrame()).empty:
        qs = results['quality_scores']
        holdings = results['holdings_history'][-1]['holdings'] if results['holdings_history'] else []
        
        if holdings and 'quality_score' in qs.columns:
            avg_quality = qs.loc[[h for h in holdings if h in qs.index], 'quality_score'].mean()
            print(f"\n    Portfolio Quality:")
            print(f"      Avg Quality Score: {avg_quality:.1f}/100")

# =============================================================================
# STEP 6: Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY: THREE-PILLAR TRANSPARENCY")
print("=" * 80)

print("""
PILLAR 1 - FUNDAMENTALS (What to Own):
  - Source: WRDS Compustat Global
  - Metrics: ROE, Debt/Equity, Profit Margin
  - Role: Filter investable universe, quality-adjust weights
  - Coverage: {} stocks matched

PILLAR 2 - REGIMES (When to Own):
  - Source: Yahoo Finance (macro), Rolling calculations (style)
  - Factors: {} time-varying series
  - Role: PLS extracts drivers, Ridge predicts returns
  - Method: Walk-forward, no look-ahead

PILLAR 3 - RISK (How Much to Own):
  - Method: Quality-weighted + inverse-vol weighting
  - Constraints: Max 10% per position
  - Rebalance: Monthly

KEY INSIGHT:
  The 'quality_weight' parameter controls the balance:
  - 0% = Pure regime/momentum strategy
  - 50% = Balanced quality + regime
  - 100% = Pure quality strategy
  
  Higher quality weight = slower turnover, fundamental anchor
  Lower quality weight = faster adaptation, regime-driven
""".format(
    len(fundamentals_df) if not fundamentals_df.empty else 0,
    len(factors.columns)
))

print("=" * 80)
print("RUN COMPLETE")
print("=" * 80)
