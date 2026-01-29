"""
End-to-end model pipeline using WRDS data

Orchestrates the complete systematic strategy:
1. Load broad, liquid European equity universe from WRDS (model decides, not judgment)
2. PLS: Compress factors into components capturing European return drivers
3. Elastic Net: Evaluate each stock's robust exposure to these drivers
4. Rank stocks by model-implied expected return
5. Construct constrained long-only portfolio from top-scoring names

Winners = stocks with most stable and positive exposure to factors driving European returns
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional

from .pls_model import PLSModel
from .elastic_net_model import ElasticNetModel
from ..data.wrds_loader_fixed import WRDSEquityUniverse, WRDSMacroFactorLoader
from ..data.style_factors import StyleFactorCalculator
from ..utils.config import (
    PLS_LOOKBACK_YEARS, ELASTIC_NET_LOOKBACK_YEARS,
    PLS_N_COMPONENTS, ELASTIC_NET_ALPHA, UNIVERSE_SIZE
)


class WRDSModelPipeline:
    """
    Complete model pipeline using WRDS data: data loading → PLS → Elastic Net → Expected Returns
    """
    
    def __init__(self):
        self.universe = WRDSEquityUniverse(top_n=UNIVERSE_SIZE)
        self.macro_loader = WRDSMacroFactorLoader()
        self.style_calculator = StyleFactorCalculator()
        self.pls_model = PLSModel(n_components=PLS_N_COMPONENTS)
        self.elastic_net_model = ElasticNetModel(alpha=ELASTIC_NET_ALPHA)
        
        self.stock_returns = None
        self.macro_factors = None
        self.style_factors = None
        self.combined_factors = None
        self.pls_components = None
        self.expected_returns = None
        
    def load_data(self, end_date: Optional[datetime] = None) -> 'WRDSModelPipeline':
        """
        Load all required data from WRDS
        
        Args:
            end_date: End date for data (default: today)
        """
        if end_date is None:
            end_date = datetime.now()
        
        print("=" * 60)
        print("LOADING DATA FROM WRDS")
        print("=" * 60)
        
        # Connect to WRDS
        self.universe.connect()
        
        # Build equity universe
        universe_df = self.universe.build_universe()
        
        # Fetch stock returns (3 years for PLS)
        pls_start = end_date - timedelta(days=PLS_LOOKBACK_YEARS * 365)
        self.stock_returns = self.universe.fetch_returns(pls_start, end_date)
        
        # Fetch fundamentals for style factors
        fundamentals = self.universe.fetch_fundamentals(end_date)
        
        # Calculate style factors from WRDS data
        self.style_factors = self._calculate_style_factors_from_wrds(
            fundamentals, self.stock_returns
        )
        
        # Load macro factors (placeholder - can be expanded)
        self.macro_factors = self.macro_loader.load_all_factors(pls_start, end_date)
        
        # Combine factors
        self.combined_factors = self._combine_factors()
        
        print("\n✅ Data loading from WRDS complete!")
        return self
    
    def _calculate_style_factors_from_wrds(self, 
                                           fundamentals: pd.DataFrame,
                                           returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate style factors using WRDS fundamental data
        """
        print("Calculating style factors from WRDS data...")
        
        factors = {}
        
        # Value factors (from fundamentals)
        if 'roe' in fundamentals.columns:
            factors['roe'] = fundamentals['roe']
        if 'debt_equity' in fundamentals.columns:
            factors['debt_equity'] = fundamentals['debt_equity']
        if 'profit_margin' in fundamentals.columns:
            factors['profit_margin'] = fundamentals['profit_margin']
        if 'revenue_growth' in fundamentals.columns:
            factors['revenue_growth'] = fundamentals['revenue_growth']
        
        # Momentum factors (from returns)
        if len(returns) >= 252:
            factors['return_1m'] = returns.tail(21).mean()
            factors['return_3m'] = returns.tail(63).mean()
            factors['return_6m'] = returns.tail(126).mean()
            factors['return_12m'] = returns.tail(252).mean()
        
        # Volatility factors
        if len(returns) >= 30:
            factors['realized_vol_30d'] = returns.tail(30).std() * np.sqrt(252)
        
        # Market cap (from fundamentals)
        if 'market_cap' in self.universe.stock_data:
            market_caps = {gvkey: data.get('market_cap', np.nan) 
                          for gvkey, data in self.universe.stock_data.items()}
            factors['market_cap'] = pd.Series(market_caps)
        
        # Combine all factors
        factors_df = pd.DataFrame(factors)
        
        # Handle missing values
        factors_df = factors_df.fillna(method='ffill').fillna(method='bfill')
        
        print(f"✅ Calculated {len(factors_df.columns)} style factors for {len(factors_df)} stocks")
        return factors_df
    
    def _combine_factors(self) -> pd.DataFrame:
        """
        Combine macro and style factors into single DataFrame
        """
        # Align with stock returns dates
        common_dates = self.stock_returns.index
        
        # Style factors are point-in-time
        style_broadcast = pd.DataFrame(
            index=common_dates,
            columns=self.style_factors.columns
        )
        for col in self.style_factors.columns:
            style_broadcast[col] = self.style_factors[col].values
        
        # If macro factors exist, combine them
        if self.macro_factors is not None and not self.macro_factors.empty:
            combined = pd.concat([self.macro_factors.loc[common_dates], style_broadcast], axis=1)
        else:
            combined = style_broadcast
        
        # Forward fill missing values
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        
        return combined
    
    def fit_pls(self) -> 'WRDSModelPipeline':
        """Fit PLS model"""
        print("\n" + "=" * 60)
        print("FITTING PLS MODEL")
        print("=" * 60)
        
        # Align factors and returns
        common_dates = self.combined_factors.index.intersection(self.stock_returns.index)
        X = self.combined_factors.loc[common_dates]
        Y = self.stock_returns.loc[common_dates]
        
        # Fit PLS
        self.pls_model.fit(X, Y)
        self.pls_components = self.pls_model.get_component_scores()
        
        print("✅ PLS model fitting complete!")
        return self
    
    def fit_elastic_net(self) -> 'WRDSModelPipeline':
        """Fit Elastic Net models"""
        print("\n" + "=" * 60)
        print("FITTING ELASTIC NET MODELS")
        print("=" * 60)
        
        # Use 1-year lookback for Elastic Net
        end_date = datetime.now()
        en_start = end_date - timedelta(days=ELASTIC_NET_LOOKBACK_YEARS * 365)
        
        # Filter to 1-year window
        pls_components_1y = self.pls_components.loc[self.pls_components.index >= en_start]
        returns_1y = self.stock_returns.loc[self.stock_returns.index >= en_start]
        
        # Fit Elastic Net
        self.elastic_net_model.fit(pls_components_1y, returns_1y)
        
        # Get expected returns using latest components
        latest_components = self.pls_components.iloc[-1:]
        expected_returns_df = self.elastic_net_model.predict(latest_components)
        self.elastic_net_model.expected_returns = expected_returns_df
        self.expected_returns = expected_returns_df['expected_return']
        
        print("✅ Elastic Net model fitting complete!")
        return self
    
    def run_full_pipeline(self, end_date: Optional[datetime] = None) -> 'WRDSModelPipeline':
        """Run complete pipeline using WRDS data"""
        try:
            self.load_data(end_date)
            self.fit_pls()
            self.fit_elastic_net()
            
            print("\n" + "=" * 60)
            print("PIPELINE COMPLETE")
            print("=" * 60)
            print(f"✅ Expected returns calculated for {len(self.expected_returns)} stocks")
            print(f"   Mean: {self.expected_returns.mean():.4f}")
            print(f"   Std: {self.expected_returns.std():.4f}")
            
            return self
        finally:
            # Clean up connections
            self.universe.close()
            self.macro_loader.close()
    
    def get_model_summary(self) -> dict:
        """Get summary of model results"""
        return {
            'n_stocks': len(self.universe.stocks),
            'n_factors': len(self.combined_factors.columns),
            'n_pls_components': self.pls_model.n_components,
            'pls_explained_variance': self.pls_model.get_explained_variance(),
            'n_elastic_net_models': len(self.elastic_net_model.models),
            'avg_r2': np.mean([
                m['r2'] for m in self.elastic_net_model.fit_metrics.values()
            ]) if self.elastic_net_model.fit_metrics else 0,
            'expected_returns_stats': {
                'mean': self.expected_returns.mean() if self.expected_returns is not None else 0,
                'std': self.expected_returns.std() if self.expected_returns is not None else 0,
                'min': self.expected_returns.min() if self.expected_returns is not None else 0,
                'max': self.expected_returns.max() if self.expected_returns is not None else 0,
            }
        }


if __name__ == "__main__":
    # Test WRDS pipeline
    pipeline = WRDSModelPipeline()
    pipeline.run_full_pipeline()
