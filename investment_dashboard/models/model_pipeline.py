"""
End-to-end model pipeline

Orchestrates the complete systematic strategy:
1. Load broad, liquid European equity universe (model decides, not judgment)
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
from ..data.equity_universe import get_equity_universe
from ..data.macro_factors import MacroFactorLoader
from ..data.style_factors import StyleFactorCalculator
from ..utils.config import (
    PLS_LOOKBACK_YEARS, ELASTIC_NET_LOOKBACK_YEARS,
    PLS_N_COMPONENTS, PLS_MIN_COMPONENTS, ELASTIC_NET_ALPHA, UNIVERSE_SIZE, DATA_SOURCE
)


class ModelPipeline:
    """
    Complete model pipeline: data loading → PLS → Elastic Net → Expected Returns
    """
    
    def __init__(self):
        # Use factory to get appropriate data loader based on config
        self.universe = get_equity_universe(top_n=UNIVERSE_SIZE)
        self.macro_loader = MacroFactorLoader()
        self.style_calculator = StyleFactorCalculator()
        self.pls_model = PLSModel(n_components=PLS_N_COMPONENTS, cv_folds=10)
        self.elastic_net_model = ElasticNetModel(
            l1_ratios=[0.01, 0.05, 0.1, 0.2, 0.3],  # Ridge-like for better signal spread
            cv_folds=10
        )
        print(f"[Config] Data source: {DATA_SOURCE}, Universe size: {UNIVERSE_SIZE}")
        
        self.stock_returns = None
        self.macro_factors = None
        self.style_factors = None
        self.combined_factors = None
        self.pls_components = None
        self.expected_returns = None
        
    def load_data(self, end_date: Optional[datetime] = None) -> 'ModelPipeline':
        """
        Load all required data
        
        Args:
            end_date: End date for data (default: today)
        """
        if end_date is None:
            end_date = datetime.now()
        
        print("=" * 60)
        print("STEP 1: LOADING DATA")
        print("=" * 60)
        
        # Step 1a: Build equity universe
        print("\n[1/5] Building equity universe (fetching stock info)...")
        print("      This fetches market cap, sector, country for ~100 stocks...")
        universe_df = self.universe.build_universe()
        print(f"      [OK] Universe built: {len(self.universe.stocks)} stocks")
        
        # Step 1b: Fetch stock returns (3 years for PLS)
        print("\n[2/5] Fetching 3 years of stock returns...")
        print("      This downloads daily prices for all stocks (takes 2-3 minutes)...")
        pls_start = end_date - timedelta(days=PLS_LOOKBACK_YEARS * 365)
        self.stock_returns = self.universe.fetch_returns(pls_start, end_date)
        
        # CRITICAL: Clean NaN values from returns
        print("      Cleaning NaN values from returns...")
        # Drop stocks with too many NaN values (>20%)
        nan_pct = self.stock_returns.isna().sum() / len(self.stock_returns)
        valid_stocks = nan_pct[nan_pct < 0.2].index.tolist()
        self.stock_returns = self.stock_returns[valid_stocks]
        # Fill remaining NaN with 0 (no return on that day)
        self.stock_returns = self.stock_returns.fillna(0)
        print(f"      [OK] Returns fetched: {len(self.stock_returns.columns)} stocks, {len(self.stock_returns)} days")
        
        # Step 1c: Fetch prices for style factors
        print("\n[3/5] Fetching prices for style factor calculation...")
        prices = self._fetch_prices(pls_start, end_date)
        # Clean prices too
        prices = prices.fillna(method='ffill').fillna(method='bfill')
        print(f"      [OK] Prices fetched: {len(prices.columns)} stocks")
        
        # Step 1d: Load macro factors
        print("\n[4/5] Loading macro factors (indices, FX, commodities)...")
        self.macro_factors = self.macro_loader.load_all_factors(pls_start, end_date)
        # Clean macro factors
        self.macro_factors = self.macro_factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
        print(f"      [OK] Macro factors loaded: {len(self.macro_factors.columns)} factors")
        
        # Step 1e: Calculate style factors
        print("\n[5/5] Calculating style factors (value, momentum, quality)...")
        tickers = [t for t in self.universe.stocks if t in self.stock_returns.columns]
        
        # If using hybrid loader, get fundamentals from WRDS (already fetched)
        wrds_fundamentals = None
        if hasattr(self.universe, 'fetch_fundamentals'):
            print("      Fetching fundamentals from WRDS (hybrid mode)...")
            wrds_fundamentals = self.universe.fetch_fundamentals()
            if wrds_fundamentals is not None and not wrds_fundamentals.empty:
                # Pass WRDS fundamentals to style calculator
                self.style_calculator.wrds_fundamentals = wrds_fundamentals
                print(f"      [OK] WRDS fundamentals loaded for {len(wrds_fundamentals)} stocks")
        
        self.style_factors = self.style_calculator.calculate_all_factors(
            tickers, prices, self.stock_returns
        )
        # Clean style factors - replace inf and NaN
        self.style_factors = self.style_factors.replace([np.inf, -np.inf], np.nan)
        self.style_factors = self.style_factors.fillna(self.style_factors.median())
        self.style_factors = self.style_factors.fillna(0)
        print(f"      [OK] Style factors calculated: {len(self.style_factors.columns)} factors")
        
        # Combine macro and style factors
        print("\n      Combining all factors...")
        self.combined_factors = self._combine_factors()
        # Final NaN clean
        self.combined_factors = self.combined_factors.fillna(method='ffill').fillna(method='bfill').fillna(0)
        print(f"      [OK] Combined factors: {len(self.combined_factors.columns)} total factors")
        
        print("\n" + "=" * 60)
        print("DATA LOADING COMPLETE!")
        print("=" * 60)
        return self
    
    def _fetch_prices(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch stock prices"""
        import yfinance as yf
        
        all_prices = []
        for ticker in self.universe.stocks:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                if not hist.empty:
                    prices = hist['Close'].copy()
                    # Remove timezone info to avoid conversion issues
                    if prices.index.tz is not None:
                        prices.index = prices.index.tz_localize(None)
                    prices.name = ticker
                    all_prices.append(prices)
            except:
                continue
        
        if all_prices:
            prices_df = pd.concat(all_prices, axis=1)
            # Ensure index is timezone-naive
            if prices_df.index.tz is not None:
                prices_df.index = prices_df.index.tz_localize(None)
            return prices_df
        else:
            raise ValueError("No price data fetched")
    
    def _combine_factors(self) -> pd.DataFrame:
        """
        Combine macro and ROLLING style factors into single DataFrame
        Style factors are now time-varying (not static snapshots)
        """
        # Ensure both indices are timezone-naive
        macro_idx = self.macro_factors.index
        returns_idx = self.stock_returns.index
        
        if hasattr(macro_idx, 'tz') and macro_idx.tz is not None:
            self.macro_factors.index = macro_idx.tz_localize(None)
        if hasattr(returns_idx, 'tz') and returns_idx.tz is not None:
            self.stock_returns.index = returns_idx.tz_localize(None)
        
        # Align macro factors with stock returns dates
        common_dates = self.macro_factors.index.intersection(self.stock_returns.index)
        macro_aligned = self.macro_factors.loc[common_dates]
        
        # Use rolling style factors (time-varying!) instead of static broadcast
        if hasattr(self, 'rolling_style_factors') and self.rolling_style_factors is not None:
            # Align rolling style factors with common dates
            style_common = self.rolling_style_factors.index.intersection(common_dates)
            style_aligned = self.rolling_style_factors.loc[style_common].reindex(common_dates)
            style_aligned = style_aligned.fillna(method='ffill').fillna(method='bfill').fillna(0)
        else:
            # Fallback: broadcast static factors (old behavior)
            style_aligned = pd.DataFrame(
                index=common_dates,
                columns=self.style_factors.columns if hasattr(self, 'style_factors') else []
            )
            if hasattr(self, 'style_factors'):
                for col in self.style_factors.columns:
                    style_aligned[col] = self.style_factors[col].iloc[-1] if len(self.style_factors[col]) > 0 else np.nan
        
        # Combine macro and style factors
        combined = pd.concat([macro_aligned, style_aligned], axis=1)
        
        # Forward fill missing values
        combined = combined.fillna(method='ffill').fillna(method='bfill')
        
        return combined
    
    def fit_pls(self, cross_validate: bool = True) -> 'ModelPipeline':
        """
        Fit PLS model to identify return-driving components
        
        PLS compresses highly correlated macro and style factors into components
        that capture the forces actually driving European equity returns.
        These components define what tends to win in Europe.
        
        Component selection via 10-fold cross-validation to minimize forecast error.
        """
        print("\n" + "=" * 60)
        print("STEP 2: FITTING PLS MODEL")
        print("=" * 60)
        print("\nPLS (Partial Least Squares) compresses factors into return-relevant components")
        print("This identifies the genuine drivers of European equity returns\n")
        
        # Align factors and returns
        print("[1/4] Aligning factor and return data...")
        common_dates = self.combined_factors.index.intersection(self.stock_returns.index)
        X = self.combined_factors.loc[common_dates].copy()
        Y = self.stock_returns.loc[common_dates].copy()
        
        # Final NaN cleaning before PLS
        print("[2/4] Final data cleaning...")
        X = X.fillna(0)
        Y = Y.fillna(0)
        
        # Remove any remaining problematic values
        X = X.replace([np.inf, -np.inf], 0)
        Y = Y.replace([np.inf, -np.inf], 0)
        
        print(f"      Data shape: {len(common_dates)} days, {len(X.columns)} factors, {len(Y.columns)} stocks")
        
        # Cross-validate to find optimal number of components
        if cross_validate:
            print("[3/4] Cross-validating PLS components (finance-aware selection)...")
            print(f"      Minimum components: {PLS_MIN_COMPONENTS} (economic factors: momentum, value, quality, size, vol)")
            optimal_n = self.pls_model.cross_validate_components(
                X, Y, 
                max_components=12, 
                min_components=PLS_MIN_COMPONENTS
            )
            print(f"      Optimal components selected: {optimal_n}")
        
        # Fit PLS with optimal components
        print("[4/4] Fitting final PLS model...")
        self.pls_model.fit(X, Y)
        
        # Get component scores
        self.pls_components = self.pls_model.get_component_scores()
        
        print(f"\n[OK] PLS complete! Extracted {self.pls_model.n_components} components")
        print(f"     Explained variance: {self.pls_model.get_explained_variance():.1%}")
        return self
    
    def fit_elastic_net(self) -> 'ModelPipeline':
        """
        Fit Elastic Net models to evaluate stock exposures
        
        For each stock, Elastic Net evaluates how strongly and consistently it benefits
        from the PLS components (return drivers). Regularization shrinks weak or unstable
        relationships to zero, retaining only robust signals. Every stock receives a
        model-implied expected return based on its stable exposure to return drivers.
        """
        print("\n" + "=" * 60)
        print("STEP 3: FITTING ELASTIC NET MODELS")
        print("=" * 60)
        print("\nElastic Net estimates each stock's exposure to return drivers")
        print("Regularization removes weak/unstable signals, keeping only robust exposures\n")
        
        # Use 1-year lookback for Elastic Net
        end_date = datetime.now()
        en_start = end_date - timedelta(days=ELASTIC_NET_LOOKBACK_YEARS * 365)
        
        # Filter to 1-year window
        print("[1/3] Filtering to 1-year lookback window...")
        pls_components_1y = self.pls_components.loc[
            self.pls_components.index >= en_start
        ].copy()
        returns_1y = self.stock_returns.loc[
            self.stock_returns.index >= en_start
        ].copy()
        
        # Clean data
        pls_components_1y = pls_components_1y.fillna(0)
        returns_1y = returns_1y.fillna(0)
        
        print(f"      Using {len(pls_components_1y)} days of data")
        
        # Fit Elastic Net (using 1-year window for training)
        print(f"[2/3] Fitting Elastic Net for {len(returns_1y.columns)} stocks...")
        print("      (This fits one model per stock - may take a minute)")
        self.elastic_net_model.fit(pls_components_1y, returns_1y)
        
        # Get expected returns using latest PLS components
        print("[3/3] Calculating expected returns...")
        latest_components = self.pls_components.iloc[-1:]
        expected_returns_df = self.elastic_net_model.predict(latest_components)
        self.elastic_net_model.expected_returns = expected_returns_df
        self.expected_returns = expected_returns_df['expected_return']
        
        print(f"\n[OK] Elastic Net complete!")
        print(f"     Models fitted for {len(self.expected_returns)} stocks")
        return self
    
    def run_full_pipeline(self, end_date: Optional[datetime] = None) -> 'ModelPipeline':
        """
        Run complete pipeline: load data → fit PLS → fit Elastic Net
        
        Args:
            end_date: End date for data (default: today)
        """
        print("\n" + "=" * 60)
        print("KING'S INVESTMENT FUND - MODEL PIPELINE")
        print("=" * 60)
        print("\nThis pipeline will:")
        print("  1. Load European equity data (~100 stocks, 3 years)")
        print("  2. Fit PLS to compress factors into return drivers")
        print("  3. Fit Elastic Net to estimate expected returns")
        print("  4. Rank stocks and build portfolio")
        print("\nEstimated time: 3-5 minutes\n")
        
        self.load_data(end_date)
        self.fit_pls()
        self.fit_elastic_net()
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE!")
        print("=" * 60)
        print(f"\nRESULTS SUMMARY:")
        print(f"   Stocks analyzed: {len(self.expected_returns)}")
        print(f"   Factors used: {len(self.combined_factors.columns)}")
        print(f"   PLS components: {self.pls_model.n_components}")
        print(f"   Mean expected return: {self.expected_returns.mean():.4f}")
        print(f"   Std expected return: {self.expected_returns.std():.4f}")
        print(f"   Top stock: {self.expected_returns.idxmax()} ({self.expected_returns.max():.4f})")
        print(f"   Bottom stock: {self.expected_returns.idxmin()} ({self.expected_returns.min():.4f})")
        
        return self
    
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
    # Test pipeline
    print("Model pipeline module loaded")
