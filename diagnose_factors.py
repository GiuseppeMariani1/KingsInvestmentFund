"""
Diagnostic script to verify what factors are actually being used
"""
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from investment_dashboard.data.style_factors import StyleFactorCalculator
from investment_dashboard.data.macro_factors import MacroFactorLoader
from investment_dashboard.data.hybrid_loader import HybridEquityUniverse

print("=" * 60)
print("FACTOR DIAGNOSTIC")
print("=" * 60)

# Build universe
universe = HybridEquityUniverse()
universe.build_universe()
print(f"\nUniverse: {len(universe.stocks)} stocks")

# Load some returns
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"\nFetching returns from {start_date.date()} to {end_date.date()}...")
returns = universe.fetch_returns(start_date, end_date)
print(f"Returns shape: {returns.shape}")

# Get prices (for momentum)
import yfinance as yf
tickers = list(returns.columns)[:50]  # Sample for speed
prices = yf.download(tickers, start=start_date, end=end_date, progress=False)['Adj Close']
prices = prices.dropna(axis=1, how='all')

# Calculate style factors
calculator = StyleFactorCalculator()
rolling_factors = calculator.calculate_rolling_style_factors(prices, returns[prices.columns])

print("\n" + "=" * 60)
print("FACTOR ANALYSIS")
print("=" * 60)

print(f"\nTotal factors: {len(rolling_factors.columns)}")
print("\nFactor columns:")
for col in sorted(rolling_factors.columns):
    series = rolling_factors[col]
    is_constant = series.std() < 1e-10
    variance = series.var()
    print(f"  {col:30s} | Var: {variance:.6f} | {'CONSTANT!' if is_constant else 'time-varying'}")

# Check which are actually time-varying
print("\n" + "=" * 60)
print("TIME-VARYING FACTORS (variance > 0.0001)")
print("=" * 60)
for col in rolling_factors.columns:
    if rolling_factors[col].var() > 0.0001:
        print(f"  {col}")

print("\n" + "=" * 60)
print("CONSTANT/NEAR-ZERO VARIANCE FACTORS")
print("=" * 60)
for col in rolling_factors.columns:
    if rolling_factors[col].var() <= 0.0001:
        val = rolling_factors[col].iloc[0]
        print(f"  {col}: constant value = {val:.4f}")

# Check quality factors specifically
print("\n" + "=" * 60)
print("QUALITY FACTOR CHECK")
print("=" * 60)
quality_cols = [c for c in rolling_factors.columns if any(x in c.lower() for x in ['roe', 'quality', 'profit', 'leverage', 'margin'])]
for col in quality_cols:
    series = rolling_factors[col]
    print(f"\n{col}:")
    print(f"  Mean: {series.mean():.6f}")
    print(f"  Std:  {series.std():.6f}")
    print(f"  Min:  {series.min():.6f}")
    print(f"  Max:  {series.max():.6f}")
    print(f"  Variance: {series.var():.10f}")
    print(f"  Is constant: {series.std() < 1e-8}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
