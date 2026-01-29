"""
Walk-Forward Backtest with Proper Out-of-Sample Testing

This implements a TRUE out-of-sample backtest:
- At each rebalance date, the model is trained ONLY on past data
- Expected returns are predicted for the NEXT period
- Portfolio is constructed and held for one period
- Realized returns are measured
- No look-ahead bias

This gives realistic, trustworthy performance metrics.
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


class WalkForwardBacktester:
    """
    Walk-forward backtester with proper out-of-sample testing
    
    At each rebalance:
    1. Train PLS + Elastic Net on data UP TO that date
    2. Predict expected returns for all stocks
    3. Select top N stocks
    4. Hold for one period, measure realized returns
    5. Roll forward and repeat
    """
    
    def __init__(self,
                 portfolio_size: int = 30,
                 rebalance_months: int = 1,
                 pls_lookback_years: int = 2,
                 pls_n_components: int = 10,
                 l1_ratios: List[float] = None):  # Kept for backwards compatibility
        """
        Args:
            portfolio_size: Number of stocks to hold
            rebalance_months: Rebalance frequency in months
            pls_lookback_years: Training window for PLS
            pls_n_components: Number of PLS components
            l1_ratios: Ignored (using pure Ridge regression)
        """
        self.portfolio_size = portfolio_size
        self.rebalance_months = rebalance_months
        self.pls_lookback_years = pls_lookback_years
        self.pls_n_components = pls_n_components
        # Ridge alphas to test (regularization strength)
        self.ridge_alphas = np.logspace(-4, 2, 30)
        
        # Results storage
        self.portfolio_returns = []
        self.benchmark_returns = []
        self.holdings_history = []
        self.model_metrics = []
        
    def run_backtest(self,
                     stock_returns: pd.DataFrame,
                     factors: pd.DataFrame,
                     backtest_start: Optional[datetime] = None,
                     backtest_end: Optional[datetime] = None,
                     progress_callback=None) -> Dict:
        """
        Run walk-forward backtest
        
        Args:
            stock_returns: Daily stock returns (dates x stocks)
            factors: Combined factor data (dates x factors)
            backtest_start: Start of backtest period (default: 1 year ago)
            backtest_end: End of backtest period (default: today)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with backtest results
        """
        print("=" * 60)
        print("WALK-FORWARD BACKTEST (True Out-of-Sample)")
        print("=" * 60)
        
        # Align data
        common_dates = stock_returns.index.intersection(factors.index)
        stock_returns = stock_returns.loc[common_dates].copy()
        factors = factors.loc[common_dates].copy()
        
        # Set backtest period
        if backtest_end is None:
            backtest_end = stock_returns.index.max()
        if backtest_start is None:
            backtest_start = backtest_end - relativedelta(years=1)
        
        # Ensure we have enough training data before backtest start
        min_train_date = backtest_start - relativedelta(years=self.pls_lookback_years)
        if stock_returns.index.min() > min_train_date:
            print(f"Warning: Limited training data, adjusting backtest start")
            backtest_start = stock_returns.index.min() + relativedelta(years=self.pls_lookback_years)
        
        print(f"Backtest period: {backtest_start.date()} to {backtest_end.date()}")
        print(f"Training window: {self.pls_lookback_years} years")
        print(f"Rebalance frequency: {self.rebalance_months} month(s)")
        print(f"Portfolio size: {self.portfolio_size} stocks")
        print("-" * 60)
        
        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(
            backtest_start, backtest_end, stock_returns.index
        )
        print(f"Number of rebalances: {len(rebalance_dates)}")
        
        # Walk forward through time
        self.portfolio_returns = []
        self.benchmark_returns = []
        self.holdings_history = []
        
        total_rebalances = len(rebalance_dates)
        
        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            next_rebal = rebalance_dates[i + 1]
            
            if progress_callback:
                progress_callback(f"Rebalance {i+1}/{total_rebalances-1}: {rebal_date.date()}")
            
            print(f"\n[{i+1}/{total_rebalances-1}] Rebalance: {rebal_date.date()}")
            
            # Get training data (ONLY past data up to rebal_date)
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
            
            # 1. Fit PLS on training data
            try:
                pls, pls_components = self._fit_pls(X_train, Y_train)
            except Exception as e:
                print(f"  PLS error: {e}")
                continue
            
            # 2. Fit Elastic Net with CV for each stock
            try:
                expected_returns = self._fit_elastic_net(pls_components, Y_train, pls)
            except Exception as e:
                print(f"  Elastic Net error: {e}")
                continue
            
            # 3. Select top stocks
            top_stocks = expected_returns.nlargest(self.portfolio_size).index.tolist()
            available_stocks = [s for s in top_stocks if s in stock_returns.columns]
            
            if len(available_stocks) < 5:
                print(f"  Skipping: not enough stocks ({len(available_stocks)})")
                continue
            
            self.holdings_history.append({
                'date': rebal_date,
                'holdings': available_stocks,
                'expected_returns': expected_returns[available_stocks].to_dict()
            })
            
            # 4. Measure realized returns in NEXT period (out-of-sample)
            hold_mask = (stock_returns.index >= rebal_date) & (stock_returns.index < next_rebal)
            period_returns = stock_returns.loc[hold_mask, available_stocks]
            
            if len(period_returns) == 0:
                continue
            
            # Equal-weighted portfolio returns
            portfolio_daily = period_returns.mean(axis=1)
            benchmark_daily = stock_returns.loc[hold_mask].mean(axis=1)
            
            self.portfolio_returns.extend(portfolio_daily.tolist())
            self.benchmark_returns.extend(benchmark_daily.tolist())
            
            # Period metrics
            port_return = (1 + portfolio_daily).prod() - 1
            bench_return = (1 + benchmark_daily).prod() - 1
            print(f"  Period return: {port_return:.2%} (benchmark: {bench_return:.2%})")
        
        # Calculate overall metrics
        results = self._calculate_results()
        
        print("\n" + "=" * 60)
        print("BACKTEST COMPLETE")
        print("=" * 60)
        print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
        print(f"Annual Return: {results['metrics']['annual_return']:.1%}")
        print(f"Annual Volatility: {results['metrics']['annual_volatility']:.1%}")
        print(f"Max Drawdown: {results['metrics']['max_drawdown']:.1%}")
        print(f"Benchmark Return: {results['metrics']['benchmark_return']:.1%}")
        
        return results
    
    def _generate_rebalance_dates(self, start: datetime, end: datetime, 
                                   available_dates: pd.DatetimeIndex) -> List[datetime]:
        """Generate monthly rebalance dates"""
        dates = []
        current = start
        
        while current <= end:
            # Find nearest available date
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
        
        # Clean
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        Y_scaled = np.nan_to_num(Y_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_components = min(self.pls_n_components, X.shape[1] - 1, X.shape[0] - 1)
        pls = PLSRegression(n_components=n_components, scale=False)
        pls.fit(X_scaled, Y_scaled)
        
        # Get component scores
        X_scores = pls.transform(X_scaled)
        pls_components = pd.DataFrame(
            X_scores,
            index=X.index,
            columns=[f'PLS_{i+1}' for i in range(n_components)]
        )
        
        # Store scaler for predictions
        pls._scaler_X = scaler_X
        
        return pls, pls_components
    
    def _fit_elastic_net(self, pls_components: pd.DataFrame, 
                         Y: pd.DataFrame, pls: PLSRegression) -> pd.Series:
        """
        Fit Elastic Net with CV for hyperparameter tuning
        Returns expected returns for all stocks
        """
        expected_returns = {}
        
        # Time series CV
        tscv = TimeSeriesSplit(n_splits=3)  # 3-fold for speed
        
        X = pls_components.values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        for stock in Y.columns:
            y = Y[stock].values
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            
            # RidgeCV - pure Ridge regression
            # Only tunes alpha (regularization strength)
            model = RidgeCV(
                alphas=np.logspace(-4, 2, 30),  # 0.0001 to 100
                cv=tscv,
                scoring='neg_mean_squared_error'
            )
            
            try:
                model.fit(X, y)
                # Predict using latest components
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
        
        # Risk-free rate
        rf_daily = 0.04 / 252
        
        # Annual metrics
        n_days = len(port_returns)
        n_years = n_days / 252
        
        total_return = (1 + port_returns).prod() - 1
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annual_vol = port_returns.std() * np.sqrt(252)
        sharpe = (annual_return - 0.04) / annual_vol if annual_vol > 0 else 0
        
        # Benchmark
        bench_total = (1 + bench_returns).prod() - 1
        bench_annual = (1 + bench_total) ** (1 / n_years) - 1 if n_years > 0 else 0
        bench_vol = bench_returns.std() * np.sqrt(252)
        bench_sharpe = (bench_annual - 0.04) / bench_vol if bench_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Sortino
        downside = port_returns[port_returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else annual_vol
        sortino = (annual_return - 0.04) / downside_vol if downside_vol > 0 else 0
        
        # Calmar
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (port_returns > 0).mean()
        
        # Information ratio
        excess = port_returns - bench_returns
        tracking_error = excess.std() * np.sqrt(252)
        ir = (annual_return - bench_annual) / tracking_error if tracking_error > 0 else 0
        
        # Beta/Alpha
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
        
        # Create time series
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


def run_walk_forward_backtest(stock_returns: pd.DataFrame,
                              factors: pd.DataFrame,
                              portfolio_size: int = 30,
                              backtest_months: int = 12,
                              progress_callback=None) -> Dict:
    """
    Convenience function for dashboard
    """
    backtester = WalkForwardBacktester(
        portfolio_size=portfolio_size,
        rebalance_months=1,
        pls_lookback_years=2,
        pls_n_components=10
    )
    
    end_date = stock_returns.index.max()
    start_date = end_date - relativedelta(months=backtest_months)
    
    results = backtester.run_backtest(
        stock_returns=stock_returns,
        factors=factors,
        backtest_start=start_date,
        backtest_end=end_date,
        progress_callback=progress_callback
    )
    
    # Add dates for index comparison
    results['start_date'] = start_date
    results['end_date'] = end_date
    
    return results


if __name__ == "__main__":
    print("Walk-forward backtest module loaded")
