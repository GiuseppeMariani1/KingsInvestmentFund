"""
THREE-PILLAR WALK-FORWARD BACKTEST
==================================
Properly integrates all three pillars:

PILLAR 1 - FUNDAMENTALS (What to own)
    - Quality filter applied BEFORE model
    - ROE, leverage, margins define investable universe
    
PILLAR 2 - REGIMES (When to own)
    - PLS extracts latent market drivers
    - Ridge regression maps to expected returns
    - Time-varying signals for regime adaptation
    
PILLAR 3 - RISK (How much to own)
    - Quality-weighted position sizing
    - Volatility scaling
    - Diversification constraints
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from .quality_filter import QualityFilter, QualityThresholds


class ThreePillarBacktester:
    """
    Walk-forward backtester with Three-Pillar integration.
    
    Key difference from basic backtest:
    - Quality filtering BEFORE stock selection
    - Quality scores influence final portfolio
    - Clear separation of concerns
    """
    
    def __init__(self,
                 portfolio_size: int = 30,
                 rebalance_months: int = 1,
                 pls_lookback_years: int = 2,
                 pls_n_components: int = 10,
                 quality_thresholds: Optional[QualityThresholds] = None,
                 quality_weight: float = 0.3):
        """
        Args:
            portfolio_size: Number of stocks to hold
            rebalance_months: Rebalance frequency in months
            pls_lookback_years: Training window for PLS
            pls_n_components: Number of PLS components
            quality_thresholds: Thresholds for quality filtering
            quality_weight: Weight of quality score in final ranking (0-1)
                           0 = pure regime signal
                           1 = pure quality ranking
        """
        self.portfolio_size = portfolio_size
        self.rebalance_months = rebalance_months
        self.pls_lookback_years = pls_lookback_years
        self.pls_n_components = pls_n_components
        self.quality_weight = quality_weight
        
        # Quality filter
        self.quality_filter = QualityFilter(quality_thresholds)
        
        # Ridge alphas
        self.ridge_alphas = np.logspace(-4, 2, 30)
        
        # Results storage
        self.portfolio_returns = []
        self.benchmark_returns = []
        self.holdings_history = []
        self.pillar_diagnostics = []
        
    def run_backtest(self,
                     stock_returns: pd.DataFrame,
                     factors: pd.DataFrame,
                     fundamentals: pd.DataFrame,
                     backtest_start: Optional[datetime] = None,
                     backtest_end: Optional[datetime] = None,
                     progress_callback=None) -> Dict:
        """
        Run walk-forward backtest with three-pillar integration.
        
        Args:
            stock_returns: Daily stock returns (dates x stocks)
            factors: Combined factor data (dates x factors) - for REGIME detection
            fundamentals: Stock fundamentals (stocks x metrics) - for QUALITY filter
            backtest_start: Start of backtest period
            backtest_end: End of backtest period
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with backtest results and pillar diagnostics
        """
        print("=" * 70)
        print("THREE-PILLAR WALK-FORWARD BACKTEST")
        print("=" * 70)
        print(f"Quality Weight: {self.quality_weight:.0%} (0%=pure regime, 100%=pure quality)")
        
        # =====================================================================
        # PILLAR 1: FUNDAMENTALS - Apply quality filter
        # =====================================================================
        print("\n" + "-" * 70)
        print("PILLAR 1: FUNDAMENTALS - Quality Filtering")
        print("-" * 70)
        
        # Calculate quality scores
        all_tickers = stock_returns.columns.tolist()
        
        if fundamentals is not None and not fundamentals.empty:
            # Filter fundamentals to stocks we have returns for
            common_tickers = [t for t in all_tickers if t in fundamentals.index]
            
            if len(common_tickers) > 0:
                quality_scores = self.quality_filter.calculate_quality_scores(
                    fundamentals.loc[common_tickers]
                )
                quality_universe, filter_stats = self.quality_filter.apply_filters(
                    fundamentals.loc[common_tickers]
                )
                
                print(f"  Total stocks: {len(all_tickers)}")
                print(f"  With fundamentals: {len(common_tickers)}")
                print(f"  Passing quality filter: {len(quality_universe)}")
                
                for filter_name, stats in filter_stats.items():
                    if isinstance(stats, dict):
                        print(f"    - {filter_name}: {stats.get('threshold', '')} "
                              f"(failed: {stats.get('failed', 0)}, remaining: {stats.get('remaining', 0)})")
            else:
                print("  WARNING: No overlap between returns and fundamentals!")
                quality_scores = pd.DataFrame()
                quality_universe = all_tickers
        else:
            print("  WARNING: No fundamentals provided - using full universe")
            quality_scores = pd.DataFrame()
            quality_universe = all_tickers
        
        # =====================================================================
        # Align data
        # =====================================================================
        common_dates = stock_returns.index.intersection(factors.index)
        stock_returns = stock_returns.loc[common_dates].copy()
        factors = factors.loc[common_dates].copy()
        
        # Set backtest period
        if backtest_end is None:
            backtest_end = stock_returns.index.max()
        if backtest_start is None:
            backtest_start = backtest_end - relativedelta(years=1)
            
        # Ensure we have enough training data
        min_train_date = backtest_start - relativedelta(years=self.pls_lookback_years)
        if stock_returns.index.min() > min_train_date:
            backtest_start = stock_returns.index.min() + relativedelta(years=self.pls_lookback_years)
        
        print("\n" + "-" * 70)
        print("PILLAR 2: REGIMES - PLS + Ridge Model")
        print("-" * 70)
        print(f"  Backtest period: {backtest_start.date()} to {backtest_end.date()}")
        print(f"  Training window: {self.pls_lookback_years} years")
        print(f"  PLS components: {self.pls_n_components}")
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(
            backtest_start, backtest_end, stock_returns.index
        )
        print(f"  Rebalance dates: {len(rebalance_dates)}")
        
        # =====================================================================
        # Walk forward through time
        # =====================================================================
        print("\n" + "-" * 70)
        print("BACKTEST EXECUTION")
        print("-" * 70)
        
        self.portfolio_returns = []
        self.benchmark_returns = []
        self.holdings_history = []
        self.pillar_diagnostics = []
        
        total_rebalances = len(rebalance_dates)
        
        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            next_rebal = rebalance_dates[i + 1]
            
            if progress_callback:
                progress_callback(f"Rebalance {i+1}/{total_rebalances-1}: {rebal_date.date()}")
            
            print(f"\n[{i+1}/{total_rebalances-1}] Rebalance: {rebal_date.date()}")
            
            # Get training data
            train_end = rebal_date
            train_start = train_end - relativedelta(years=self.pls_lookback_years)
            
            train_mask = (stock_returns.index >= train_start) & (stock_returns.index < train_end)
            X_train = factors.loc[train_mask].copy()
            Y_train = stock_returns.loc[train_mask].copy()
            
            # Clean training data
            X_train = X_train.fillna(0).replace([np.inf, -np.inf], 0)
            Y_train = Y_train.fillna(0).replace([np.inf, -np.inf], 0)
            
            if len(X_train) < 100:
                print(f"  Skipping: insufficient training data ({len(X_train)} days)")
                continue
            
            # =================================================================
            # PILLAR 2: Fit PLS + Ridge model (REGIME)
            # =================================================================
            try:
                pls, pls_components = self._fit_pls(X_train, Y_train)
                regime_scores = self._fit_ridge(pls_components, Y_train, pls)
            except Exception as e:
                print(f"  Model error: {e}")
                continue
            
            # =================================================================
            # COMBINE PILLARS: Quality + Regime
            # =================================================================
            combined_scores = self._combine_pillar_scores(
                regime_scores, 
                quality_scores,
                quality_universe
            )
            
            # Select top stocks from combined score
            top_stocks = combined_scores.nlargest(self.portfolio_size).index.tolist()
            available_stocks = [s for s in top_stocks if s in stock_returns.columns]
            
            if len(available_stocks) < 5:
                print(f"  Skipping: not enough stocks ({len(available_stocks)})")
                continue
            
            # =================================================================
            # PILLAR 3: Position sizing (RISK)
            # =================================================================
            weights = self._calculate_position_weights(
                available_stocks, 
                quality_scores,
                stock_returns.loc[:rebal_date, available_stocks]
            )
            
            # Store diagnostics
            diagnostics = {
                'date': rebal_date,
                'quality_universe_size': len(quality_universe),
                'regime_signal_range': regime_scores.max() - regime_scores.min(),
                'combined_score_range': combined_scores.max() - combined_scores.min(),
                'avg_quality_score': quality_scores.loc[available_stocks, 'quality_score'].mean() 
                    if not quality_scores.empty and 'quality_score' in quality_scores.columns else np.nan,
                'max_weight': max(weights.values()) if weights else 0,
                'min_weight': min(weights.values()) if weights else 0,
            }
            self.pillar_diagnostics.append(diagnostics)
            
            self.holdings_history.append({
                'date': rebal_date,
                'holdings': available_stocks,
                'weights': weights,
                'regime_scores': regime_scores[available_stocks].to_dict(),
                'quality_scores': quality_scores.loc[available_stocks, 'quality_score'].to_dict()
                    if not quality_scores.empty and 'quality_score' in quality_scores.columns 
                    else {}
            })
            
            # =================================================================
            # Measure realized returns (out-of-sample)
            # =================================================================
            hold_mask = (stock_returns.index >= rebal_date) & (stock_returns.index < next_rebal)
            period_returns = stock_returns.loc[hold_mask, available_stocks]
            
            if len(period_returns) == 0:
                continue
            
            # Weighted portfolio returns
            weight_series = pd.Series(weights)
            portfolio_daily = (period_returns * weight_series).sum(axis=1)
            benchmark_daily = stock_returns.loc[hold_mask].mean(axis=1)
            
            self.portfolio_returns.extend(portfolio_daily.tolist())
            self.benchmark_returns.extend(benchmark_daily.tolist())
            
            # Period metrics
            port_return = (1 + portfolio_daily).prod() - 1
            bench_return = (1 + benchmark_daily).prod() - 1
            
            print(f"  Holdings: {len(available_stocks)} stocks | "
                  f"Return: {port_return:.2%} (bench: {bench_return:.2%})")
        
        # Calculate overall metrics
        results = self._calculate_results()
        
        # Add pillar diagnostics
        results['pillar_diagnostics'] = pd.DataFrame(self.pillar_diagnostics)
        results['quality_filter_stats'] = self.quality_filter.filter_stats
        results['quality_scores'] = quality_scores
        
        print("\n" + "=" * 70)
        print("THREE-PILLAR BACKTEST COMPLETE")
        print("=" * 70)
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Annual Return: {results['metrics']['annual_return']:.1%}")
        print(f"Max Drawdown: {results['metrics']['max_drawdown']:.1%}")
        print(f"Benchmark: {results['metrics']['benchmark_return']:.1%}")
        
        return results
    
    def _combine_pillar_scores(self, 
                               regime_scores: pd.Series,
                               quality_scores: pd.DataFrame,
                               quality_universe: List[str]) -> pd.Series:
        """
        Combine regime signal with quality score.
        
        Final Score = (1 - quality_weight) * regime_rank + quality_weight * quality_rank
        """
        # Start with regime scores
        combined = pd.Series(index=regime_scores.index, dtype=float)
        
        # Normalize regime scores to [0, 1] rank
        regime_rank = regime_scores.rank(pct=True)
        
        if quality_scores.empty or 'quality_score' not in quality_scores.columns:
            # No quality data - use pure regime
            combined = regime_rank
        else:
            # Get quality ranks
            quality_rank = quality_scores['quality_score'].rank(pct=True)
            
            # Combine
            for ticker in regime_scores.index:
                regime_r = regime_rank.get(ticker, 0.5)
                
                if ticker in quality_rank.index:
                    quality_r = quality_rank[ticker]
                else:
                    quality_r = 0.5  # Neutral if missing
                
                # Weighted combination
                combined[ticker] = (
                    (1 - self.quality_weight) * regime_r + 
                    self.quality_weight * quality_r
                )
        
        # Apply quality universe filter - heavily penalize stocks not in quality universe
        for ticker in combined.index:
            if ticker not in quality_universe:
                combined[ticker] *= 0.1  # 90% penalty for non-quality stocks
        
        return combined
    
    def _calculate_position_weights(self,
                                    stocks: List[str],
                                    quality_scores: pd.DataFrame,
                                    historical_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate position weights using Pillar 3 (Risk).
        
        Weights based on:
        1. Quality score (higher quality = larger position)
        2. Inverse volatility (lower vol = larger position)
        3. Equal weight baseline
        """
        n_stocks = len(stocks)
        base_weight = 1.0 / n_stocks
        
        weights = {}
        
        # Get recent volatilities (30-day)
        recent_returns = historical_returns.tail(30)
        volatilities = recent_returns.std()
        avg_vol = volatilities.mean()
        
        for stock in stocks:
            weight = base_weight
            
            # Quality adjustment
            if not quality_scores.empty and stock in quality_scores.index:
                q_score = quality_scores.loc[stock, 'quality_score']
                # Scale: 50 = 1.0x, 100 = 1.25x, 0 = 0.75x
                quality_mult = 0.75 + (q_score / 100) * 0.5
                weight *= quality_mult
            
            # Volatility adjustment (inverse vol weighting)
            if stock in volatilities.index and volatilities[stock] > 0:
                vol_mult = avg_vol / volatilities[stock]
                # Cap adjustment
                vol_mult = np.clip(vol_mult, 0.5, 2.0)
                weight *= vol_mult
            
            weights[stock] = weight
        
        # Normalize to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        # Apply max position constraint (10% max)
        max_weight = 0.10
        weights = {k: min(v, max_weight) for k, v in weights.items()}
        
        # Renormalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _generate_rebalance_dates(self, start: datetime, end: datetime, 
                                   available_dates: pd.DatetimeIndex) -> List[datetime]:
        """Generate monthly rebalance dates"""
        dates = []
        current = start
        
        while current <= end:
            mask = available_dates >= current
            if mask.any():
                nearest = available_dates[mask][0]
                if nearest <= end:
                    dates.append(nearest)
            current = current + relativedelta(months=self.rebalance_months)
        
        return dates
    
    def _fit_pls(self, X: pd.DataFrame, Y: pd.DataFrame) -> Tuple[PLSRegression, pd.DataFrame]:
        """Fit PLS model on training data"""
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X.values)
        Y_scaled = scaler_Y.fit_transform(Y.values)
        
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        Y_scaled = np.nan_to_num(Y_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_components = min(self.pls_n_components, X.shape[1] - 1, X.shape[0] - 1)
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_scaled, Y_scaled)
        
        X_scores = pls.transform(X_scaled)
        pls_components = pd.DataFrame(
            X_scores,
            index=X.index,
            columns=[f'PLS_{i+1}' for i in range(n_components)]
        )
        
        pls._scaler_X = scaler_X
        return pls, pls_components
    
    def _fit_ridge(self, pls_components: pd.DataFrame, 
                   Y: pd.DataFrame, pls: PLSRegression) -> pd.Series:
        """Fit Ridge model for each stock, return expected returns"""
        expected_returns = {}
        
        tscv = TimeSeriesSplit(n_splits=3)
        X = pls_components.values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        for stock in Y.columns:
            y = Y[stock].values
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            model = RidgeCV(
                alphas=self.ridge_alphas,
                cv=tscv,
                scoring='neg_mean_squared_error'
            )
            
            try:
                model.fit(X, y)
                expected_returns[stock] = model.predict(X[-1:].reshape(1, -1))[0]
            except:
                expected_returns[stock] = 0.0
        
        return pd.Series(expected_returns)
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results and metrics"""
        if not self.portfolio_returns:
            return {'metrics': {}, 'portfolio_returns': pd.Series(), 
                    'benchmark_returns': pd.Series()}
        
        port_returns = pd.Series(self.portfolio_returns)
        bench_returns = pd.Series(self.benchmark_returns)
        
        # Metrics calculation (same as before)
        n_days = len(port_returns)
        n_years = n_days / 252
        
        total_return = (1 + port_returns).prod() - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.04) / annual_vol if annual_vol > 0 else 0
        
        bench_total = (1 + bench_returns).prod() - 1
        bench_annual = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 else 0
        bench_vol = bench_returns.std() * np.sqrt(252)
        bench_sharpe = (bench_annual - 0.04) / bench_vol if bench_vol > 0 else 0
        
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        downside = port_returns[port_returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else annual_vol
        sortino = (annual_return - 0.04) / downside_vol if downside_vol > 0 else 0
        
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        win_rate = (port_returns > 0).mean()
        
        excess = port_returns - bench_returns
        tracking_error = excess.std() * np.sqrt(252)
        ir = (annual_return - bench_annual) / tracking_error if tracking_error > 0 else 0
        
        if len(port_returns) > 30:
            cov = np.cov(port_returns.values, bench_returns.values)
            beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1
            alpha = annual_return - (0.04 + beta * (bench_annual - 0.04))
        else:
            beta = 1
            alpha = annual_return - bench_annual
        
        metrics = {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': ir,
            'tracking_error': tracking_error,
            'benchmark_return': bench_annual,
            'benchmark_volatility': bench_vol,
            'benchmark_sharpe': bench_sharpe,
            'excess_return': annual_return - bench_annual,
            'n_days': n_days,
            'n_years': n_years,
            'n_rebalances': len(self.holdings_history)
        }
        
        port_cum = (1 + port_returns).cumprod()
        bench_cum = (1 + bench_returns).cumprod()
        
        return {
            'metrics': metrics,
            'portfolio_returns': port_returns,
            'benchmark_returns': bench_returns,
            'portfolio_cumulative': port_cum,
            'benchmark_cumulative': bench_cum,
            'drawdowns': drawdowns,
            'holdings_history': self.holdings_history
        }


def run_three_pillar_backtest(stock_returns: pd.DataFrame,
                              factors: pd.DataFrame,
                              fundamentals: pd.DataFrame,
                              portfolio_size: int = 30,
                              backtest_months: int = 12,
                              quality_weight: float = 0.3,
                              progress_callback=None) -> Dict:
    """
    Convenience function for three-pillar backtest.
    
    Args:
        stock_returns: Daily returns (dates x stocks)
        factors: Regime factors for PLS (dates x factors)
        fundamentals: Stock fundamentals (stocks x metrics)
        portfolio_size: Number of stocks
        backtest_months: Backtest duration
        quality_weight: 0=pure regime, 1=pure quality
        progress_callback: Progress callback
    """
    backtester = ThreePillarBacktester(
        portfolio_size=portfolio_size,
        rebalance_months=1,
        pls_lookback_years=2,
        pls_n_components=10,
        quality_weight=quality_weight
    )
    
    end_date = stock_returns.index.max()
    start_date = end_date - relativedelta(months=backtest_months)
    
    results = backtester.run_backtest(
        stock_returns=stock_returns,
        factors=factors,
        fundamentals=fundamentals,
        backtest_start=start_date,
        backtest_end=end_date,
        progress_callback=progress_callback
    )
    
    results['start_date'] = start_date
    results['end_date'] = end_date
    
    return results
