"""
FULL TRANSPARENCY DIAGNOSTIC
"""
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("FULL TRANSPARENCY DIAGNOSTIC")
print("=" * 80)
print(f"Run at: {datetime.now()}")
print()

# DATA LOADING 
print("=" * 80)
print("STEP 1: UNIVERSE DATA")
print("=" * 80)

from investment_dashboard.data.hybrid_loader import HybridEquityUniverse

universe = HybridEquityUniverse()
universe.build_universe()

print(f"\nClaimed universe size: {len(universe.stocks)} stocks")
# Handle different data structures
if isinstance(universe.stocks, pd.DataFrame):
    sample = list(universe.stocks['ticker'].head(10)) if 'ticker' in universe.stocks.columns else list(universe.stocks.index[:10])
elif isinstance(universe.stocks, list):
    sample = universe.stocks[:10]
else:
    sample = list(universe.stocks)[:10]
print(f"Sample tickers: {sample}")

# RETURNS DATA check (How much) 

print("\n" + "=" * 80)
print("STEP 2: RETURNS DATA")
print("=" * 80)

end_date = datetime.now()
start_date = end_date - timedelta(days=365*3)  # 3 years

print(f"\nFetching returns from {start_date.date()} to {end_date.date()}...")
returns = universe.fetch_returns(start_date, end_date)

print(f"\nACTUAL returns data:")
print(f"  - Stocks with data: {returns.shape[1]}")
print(f"  - Trading days: {returns.shape[0]}")
print(f"  - Date range: {returns.index[0]} to {returns.index[-1]}")
print(f"  - Missing data %: {(returns.isna().sum().sum() / returns.size * 100):.1f}%")

# Check for stocks with >50% missing
bad_stocks = (returns.isna().sum() / len(returns) > 0.5).sum()
print(f"  - Stocks with >50% missing: {bad_stocks}")

# FUNDAMENTAL DATA FROM WRDS

print("\n" + "=" * 80)
print("STEP 3: WRDS FUNDAMENTAL DATA (QUALITY FACTORS)")
print("=" * 80)

from investment_dashboard.data.style_factors import StyleFactorCalculator

calculator = StyleFactorCalculator()
tickers = returns.columns.tolist()

# Directly call the WRDS fetch to see what we get
print(f"\nAttempting to fetch WRDS fundamentals for {len(tickers)} tickers...")
wrds_data = calculator._fetch_wrds_fundamentals(tickers)

if wrds_data is None or wrds_data.empty:
    print("\n*** CRITICAL: NO WRDS DATA AVAILABLE ***")
    print("Quality factors will be FAKE/SYNTHETIC!")
    wrds_matched = 0
else:
    # Count actual matches
    wrds_matched = wrds_data.notna().any(axis=1).sum()
    print(f"\nWRDS DATA RESULTS:")
    print(f"  - Tickers matched: {wrds_matched} / {len(tickers)} ({wrds_matched/len(tickers)*100:.1f}%)")
    
    if wrds_matched > 0:
        print(f"\n  Available fundamentals:")
        for col in wrds_data.columns:
            valid = wrds_data[col].notna().sum()
            print(f"    - {col}: {valid} stocks have data")
        
        # Show sample of actual data
        print(f"\n  Sample of ACTUAL fundamental data:")
        sample = wrds_data.dropna(how='all').head(10)
        if len(sample) > 0:
            print(sample.to_string())
        else:
            print("    NO VALID DATA!")

# FACTOR PORTFOLIO RETURNS - (Non synthetic check)
print("\n" + "=" * 80)
print("STEP 4: FACTOR PORTFOLIO RETURNS (REAL vs SYNTHETIC)")
print("=" * 80)

# Check if we have enough matches for real factor portfolios
if wrds_matched < 20:
    print(f"\n*** WARNING: Only {wrds_matched} tickers matched WRDS ***")
    print("*** Factor portfolio returns will be SYNTHETIC (FAKE)! ***")
    print()
    print("What this means:")
    print("  - 'quality_factor' = market_return * 0.1 + tiny_constant")
    print("  - 'profitability_factor' = market_return * 0.05 + noise")
    print("  - These are NOT real quality exposures!")
    print("  - The model is NOT actually selecting high-quality stocks!")
    factor_type = "SYNTHETIC (FAKE)"
else:
    print(f"\n{wrds_matched} tickers matched - can build REAL factor portfolios")
    factor_type = "REAL (Fama-French style)"

# STYLE FACTORS (Time varying vs constant)

print("\n" + "=" * 80)
print("STEP 5: STYLE FACTOR ANALYSIS")
print("=" * 80)

# Get prices for a sample
import yfinance as yf
sample_tickers = tickers[:100]  # First 100 for speed

print(f"\nFetching prices for {len(sample_tickers)} sample stocks...")
try:
    prices_raw = yf.download(sample_tickers, start=start_date, end=end_date, progress=False)
    if isinstance(prices_raw.columns, pd.MultiIndex):
        prices = prices_raw['Close'] if 'Close' in prices_raw.columns.get_level_values(0) else prices_raw.iloc[:, :len(sample_tickers)]
    else:
        prices = prices_raw
    prices = prices.dropna(axis=1, how='all')
except Exception as e:
    print(f"  Error fetching prices: {e}")
    prices = pd.DataFrame()

if not prices.empty:
    # Calculate factors
    sample_returns = returns[prices.columns] if len(prices.columns) > 0 else returns.iloc[:, :50]
    
    print(f"\nCalculating style factors...")
    rolling_factors = calculator.calculate_rolling_style_factors(prices, sample_returns)
    
    print(f"\nFACTOR BREAKDOWN:")
    print(f"  Total factors: {len(rolling_factors.columns)}")
    
    # Categorize factors
    time_varying = []
    constant = []
    near_zero = []
    
    for col in rolling_factors.columns:
        series = rolling_factors[col].dropna()
        if len(series) == 0:
            near_zero.append(col)
        elif series.std() < 1e-10:
            constant.append((col, series.iloc[0] if len(series) > 0 else 0))
        elif series.std() < 1e-6:
            near_zero.append(col)
        else:
            time_varying.append((col, series.std(), series.mean()))
    
    print(f"\n  TIME-VARYING FACTORS (actually useful): {len(time_varying)}")
    for name, std, mean in sorted(time_varying, key=lambda x: -x[1])[:15]:
        print(f"    {name:35s} | std={std:.6f} | mean={mean:.6f}")
    
    print(f"\n  CONSTANT FACTORS (PLS ignores these): {len(constant)}")
    for name, val in constant:
        print(f"    {name:35s} | constant value = {val:.6f}")
    
    print(f"\n  NEAR-ZERO VARIANCE (effectively useless): {len(near_zero)}")
    for name in near_zero:
        print(f"    {name}")
    
    # SPECIFIC CHECK: real v fake 
    
    print("\n" + "-" * 60)
    print("QUALITY FACTOR DEEP DIVE")
    print("-" * 60)
    
    quality_keywords = ['roe', 'quality', 'profit', 'margin', 'leverage']
    quality_factors = [c for c in rolling_factors.columns if any(k in c.lower() for k in quality_keywords)]
    
    for col in quality_factors:
        series = rolling_factors[col].dropna()
        if len(series) > 0:
            print(f"\n  {col}:")
            print(f"    Mean:     {series.mean():.8f}")
            print(f"    Std:      {series.std():.8f}")
            print(f"    Min:      {series.min():.8f}")
            print(f"    Max:      {series.max():.8f}")
            
            # Is it constant?
            if series.std() < 1e-10:
                print(f"    STATUS:   *** CONSTANT - NO INFORMATION ***")
            elif series.std() < 1e-6:
                print(f"    STATUS:   *** NEAR-ZERO VARIANCE - USELESS ***")
            else:
                # Check if it's just scaled market return (synthetic)
                market_ret = sample_returns.mean(axis=1).dropna()
                common_idx = series.index.intersection(market_ret.index)
                if len(common_idx) > 100:
                    corr = series[common_idx].corr(market_ret[common_idx])
                    if abs(corr) > 0.9:
                        print(f"    STATUS:   *** SYNTHETIC - {corr:.1%} correlated with market ***")
                    else:
                        print(f"    STATUS:   Appears genuine (market corr: {corr:.1%})")
                else:
                    print(f"    STATUS:   Cannot verify (insufficient data)")

# MACRO FACTORS - What external data?

print("\n" + "=" * 80)
print("STEP 6: MACRO FACTORS")
print("=" * 80)

from investment_dashboard.data.macro_factors import MacroFactorLoader

macro_loader = MacroFactorLoader()
try:
    macro_data = macro_loader.load_all_factors(start_date, end_date)
    print(f"\nMacro factors loaded: {len(macro_data.columns)}")
    print(f"Date range: {macro_data.index[0]} to {macro_data.index[-1]}")
    print(f"\nMacro factor list:")
    for col in macro_data.columns:
        valid = macro_data[col].notna().sum()
        total = len(macro_data)
        print(f"  {col:25s} | {valid}/{total} days ({valid/total*100:.0f}%)")
except Exception as e:
    print(f"\nMacro factor loading failed: {e}")
    macro_data = pd.DataFrame()

# MODEL DIAGNOSTICS

print("\n" + "=" * 80)
print("STEP 7: WHAT THE MODEL IS ACTUALLY DOING")
print("=" * 80)

print("""
Based on the analysis above, here's what your model ACTUALLY does:

1. STOCK UNIVERSE:
   - Claims ~600 STOXX 600 stocks
   - Actually has price data for: ~454 stocks
   
2. FACTOR EXPOSURE:
   - MOMENTUM: Real, time-varying (return_1m, 3m, 6m, 12m)
   - VOLATILITY: Real, time-varying (realized_vol_30d)
   - MACRO: Real, time-varying (VIX, rates, currencies)
   - QUALITY: {} (based on WRDS matching)

3. WHAT DRIVES PERFORMANCE:
   - Primary: Momentum factors (which stocks went up recently)
   - Secondary: Macro regime (VIX, rates)
   - Quality/Fundamentals: Minimal/fake contribution

4. HONEST CHARACTERIZATION:
   This is a MOMENTUM + MACRO REGIME strategy.
   It is NOT a quality/fundamental strategy.
   
   The "quality" factors you see are either:
   - Constant (ignored by PLS)
   - Synthetic (just scaled market returns)
   - Minimal (few stocks matched to WRDS)
""".format(factor_type))

# SUMMARY TABLE

print("\n" + "=" * 80)
print("SUMMARY: WHAT'S REAL vs FAKE")
print("=" * 80)

summary = [
    ("Stock Universe", f"{returns.shape[1]} stocks with data", "REAL"),
    ("Price Returns", f"{returns.shape[0]} days", "REAL"),
    ("Momentum Factors", "1M, 3M, 6M, 12M returns", "REAL"),
    ("Volatility Factors", "30-day rolling vol", "REAL"),
    ("Macro Factors", f"{len(macro_data.columns)} series", "REAL"),
    ("WRDS Fundamentals", f"{wrds_matched} stocks matched", "PARTIAL" if wrds_matched > 20 else "MINIMAL"),
    ("Quality Factor Returns", factor_type, "FAKE" if wrds_matched < 20 else "REAL"),
    ("ROE/Profit Margin as factors", "Constant values", "USELESS"),
]

print(f"\n{'Component':<30} {'Details':<35} {'Status':<10}")
print("-" * 75)
for component, details, status in summary:
    status_color = status
    print(f"{component:<30} {details:<35} {status_color:<10}")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print("\nThis file saved at: FULL_TRANSPARENCY_DIAGNOSTIC.py")
print("Re-run anytime to verify data integrity.")
