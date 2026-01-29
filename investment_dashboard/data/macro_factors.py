"""
Macro Factor Data Loading
Fetches macroeconomic factors from Yahoo Finance
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import MACRO_FACTORS


class MacroFactorLoader:
    """Loads macroeconomic factors"""
    
    def __init__(self):
        self.factors = {}
        self.factor_data = {}
    
    def fetch_factor(self, ticker: str, start_date: datetime, end_date: datetime) -> Optional[pd.Series]:
        """Fetch a single macro factor"""
        try:
            data = yf.Ticker(ticker)
            hist = data.history(start=start_date, end=end_date)
            
            if hist.empty:
                return None
            
            # Use Close price
            series = hist['Close'].copy()
            # Remove timezone info to avoid conversion issues
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            series.name = ticker
            return series
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None
    
    def load_all_factors(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load all macro factors and return as DataFrame
        
        Returns:
            DataFrame with columns: date, factor_name, value
            Pivoted to wide format: dates x factors
        """
        print("Loading macro factors...")
        
        all_factors = []
        
        for category, factors in MACRO_FACTORS.items():
            for factor_name, ticker in factors.items():
                series = self.fetch_factor(ticker, start_date, end_date)
                if series is not None:
                    df = pd.DataFrame({
                        'date': series.index,
                        'factor': factor_name,
                        'value': series.values
                    })
                    all_factors.append(df)
                    self.factor_data[factor_name] = series
                else:
                    print(f"Warning: Could not fetch {factor_name} ({ticker})")
        
        if not all_factors:
            raise ValueError("No macro factors loaded.")
        
        # Combine and pivot
        factors_df = pd.concat(all_factors, ignore_index=True)
        # Ensure dates are timezone-naive
        factors_df['date'] = pd.to_datetime(factors_df['date']).dt.tz_localize(None)
        
        # Pivot to wide format
        factors_wide = factors_df.pivot(index='date', columns='factor', values='value')
        
        # Forward fill missing values (macro factors update less frequently)
        factors_wide = factors_wide.fillna(method='ffill').fillna(method='bfill')
        
        print(f"Loaded {len(factors_wide.columns)} macro factors over {len(factors_wide)} days")
        return factors_wide
    
    def calculate_factor_returns(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns/changes for macro factors
        Some factors are levels (prices), some are rates (already in %)
        """
        factor_returns = factors_df.pct_change().dropna()
        return factor_returns
    
    def get_factor_summary(self) -> pd.DataFrame:
        """Get summary statistics of loaded factors"""
        summary = []
        for factor_name, series in self.factor_data.items():
            summary.append({
                'factor': factor_name,
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'last_value': series.iloc[-1],
                'data_points': len(series)
            })
        return pd.DataFrame(summary)


if __name__ == "__main__":
    # Test macro factor loading
    loader = MacroFactorLoader()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    
    factors = loader.load_all_factors(start_date, end_date)
    print("\nMacro Factors:")
    print(factors.head())
    print(f"\nFactor columns: {list(factors.columns)}")
