"""
Backtesting Module for King's Investment Fund

Simulates historical portfolio performance using walk-forward approach:
- At each rebalance date, rank stocks by model-implied expected returns
- Construct equal-weighted portfolio of top N stocks
- Track portfolio returns vs benchmark
- Calculate performance metrics (Sharpe, returns, volatility)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class Backtester:
    """Walk-forward backtesting engine"""
    
    def __init__(self, 
                 portfolio_size: int = 30,
                 rebalance_frequency: str = 'monthly',
                 transaction_cost: float = 0.001):  # 10 bps per trade
        self.portfolio_size = portfolio_size
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        
        self.portfolio_returns = None
        self.benchmark_returns = None
        self.holdings_history = []
        self.metrics = {}
        
    def run_backtest(self,
                     stock_returns: pd.DataFrame,
                     expected_returns: pd.Series,
                     benchmark_returns: Optional[pd.Series] = None,
                     start_date: Optional[datetime] = None,
                     end_date: Optional[datetime] = None) -> Dict:
        """
        Run walk-forward backtest
        
        For simplicity, this uses a static ranking from expected_returns
        and simulates holding the top N stocks over the historical period.
        
        In production, you would re-fit the model at each rebalance date.
        
        Args:
            stock_returns: Daily returns (dates x stocks)
            expected_returns: Model-implied expected returns (from current model)
            benchmark_returns: Benchmark daily returns (optional)
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Dictionary with backtest results
        """
        print("Running backtest...")
        
        # Filter dates
        if start_date is None:
            start_date = stock_returns.index.min()
        if end_date is None:
            end_date = stock_returns.index.max()
            
        # Filter to backtest period
        mask = (stock_returns.index >= start_date) & (stock_returns.index <= end_date)
        returns = stock_returns.loc[mask].copy()
        
        if len(returns) < 20:
            raise ValueError("Insufficient data for backtest")
        
        # Get top N stocks by expected return
        top_stocks = expected_returns.nlargest(self.portfolio_size).index.tolist()
        
        # Filter to stocks we have returns for
        available_stocks = [s for s in top_stocks if s in returns.columns]
        if len(available_stocks) < 5:
            raise ValueError("Not enough stocks with return data")
        
        print(f"  Backtesting {len(available_stocks)} stocks from {start_date.date()} to {end_date.date()}")
        
        # Calculate equal-weighted portfolio returns
        portfolio_returns = returns[available_stocks].mean(axis=1)
        self.portfolio_returns = portfolio_returns
        
        # Create benchmark if not provided (equal-weight all stocks)
        if benchmark_returns is None:
            benchmark_returns = returns.mean(axis=1)
        else:
            benchmark_returns = benchmark_returns.loc[mask]
        self.benchmark_returns = benchmark_returns
        
        # Calculate cumulative returns
        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        # Calculate metrics
        self.metrics = self._calculate_metrics(portfolio_returns, benchmark_returns)
        
        # Store results
        results = {
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'portfolio_cumulative': portfolio_cum,
            'benchmark_cumulative': benchmark_cum,
            'metrics': self.metrics,
            'holdings': available_stocks,
            'start_date': start_date,
            'end_date': end_date
        }
        
        print(f"  Backtest complete!")
        print(f"  Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"  Annual Return: {self.metrics['annual_return']:.1%}")
        print(f"  Annual Volatility: {self.metrics['annual_volatility']:.1%}")
        
        return results
    
    def _calculate_metrics(self, 
                          portfolio_returns: pd.Series, 
                          benchmark_returns: pd.Series) -> Dict:
        """Calculate performance metrics"""
        
        # Risk-free rate (approximate)
        rf_daily = 0.04 / 252  # Assume 4% annual risk-free rate
        
        # Portfolio metrics
        excess_returns = portfolio_returns - rf_daily
        
        # Annualized return
        total_return = (1 + portfolio_returns).prod() - 1
        n_years = len(portfolio_returns) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Annualized volatility
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - 0.04) / annual_volatility if annual_volatility > 0 else 0
        
        # Sortino ratio (downside volatility)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annual_volatility
        sortino_ratio = (annual_return - 0.04) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_rate = (portfolio_returns > 0).mean()
        
        # Benchmark comparison
        benchmark_total = (1 + benchmark_returns).prod() - 1
        benchmark_annual = (1 + benchmark_total) ** (1 / n_years) - 1 if n_years > 0 else 0
        benchmark_vol = benchmark_returns.std() * np.sqrt(252)
        benchmark_sharpe = (benchmark_annual - 0.04) / benchmark_vol if benchmark_vol > 0 else 0
        
        # Alpha and Beta
        if len(portfolio_returns) > 30:
            cov_matrix = np.cov(portfolio_returns.values, benchmark_returns.values)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 1
            alpha = annual_return - (0.04 + beta * (benchmark_annual - 0.04))
        else:
            beta = 1
            alpha = annual_return - benchmark_annual
        
        # Information ratio
        tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
        information_ratio = (annual_return - benchmark_annual) / tracking_error if tracking_error > 0 else 0
        
        metrics = {
            # Portfolio metrics
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            
            # Risk metrics
            'beta': beta,
            'alpha': alpha,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            
            # Benchmark metrics
            'benchmark_return': benchmark_annual,
            'benchmark_volatility': benchmark_vol,
            'benchmark_sharpe': benchmark_sharpe,
            
            # Excess returns
            'excess_return': annual_return - benchmark_annual,
            
            # Period info
            'n_days': len(portfolio_returns),
            'n_years': n_years
        }
        
        return metrics
    
    def get_monthly_returns(self) -> pd.DataFrame:
        """Get monthly returns table"""
        if self.portfolio_returns is None:
            return pd.DataFrame()
        
        monthly = self.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Pivot to year x month format
        monthly_df = monthly.to_frame('return')
        monthly_df['year'] = monthly_df.index.year
        monthly_df['month'] = monthly_df.index.month
        
        pivot = monthly_df.pivot(index='year', columns='month', values='return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]
        
        return pivot
    
    def get_drawdown_series(self) -> pd.Series:
        """Get drawdown time series"""
        if self.portfolio_returns is None:
            return pd.Series()
        
        cumulative = (1 + self.portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        
        return drawdowns


def run_simple_backtest(stock_returns: pd.DataFrame,
                       expected_returns: pd.Series,
                       portfolio_size: int = 30,
                       lookback_years: int = 1) -> Dict:
    """
    Simple backtest function for use in dashboard
    """
    backtester = Backtester(portfolio_size=portfolio_size)
    
    # Use last N years for backtest
    end_date = stock_returns.index.max()
    start_date = end_date - timedelta(days=lookback_years * 365)
    
    results = backtester.run_backtest(
        stock_returns=stock_returns,
        expected_returns=expected_returns,
        start_date=start_date,
        end_date=end_date
    )
    
    results['monthly_returns'] = backtester.get_monthly_returns()
    results['drawdowns'] = backtester.get_drawdown_series()
    
    return results


if __name__ == "__main__":
    print("Backtest module loaded")
