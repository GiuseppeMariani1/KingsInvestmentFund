"""
Portfolio Optimization Module

Academic Background:
====================
This module implements Mean-Variance Optimization (MVO) enhanced with GARCH(1,1)
volatility forecasts. The approach builds on Markowitz's (1952) Modern Portfolio Theory
while addressing its key weakness: static volatility estimates.

Mean-Variance Optimization:
--------------------------
Markowitz showed that investors should maximize:

    max  w'μ - (λ/2) w'Σw
    s.t. w'1 = 1, w ≥ 0 (long-only)

Or equivalently, maximize the Sharpe Ratio:

    max  (w'μ - rf) / √(w'Σw)

Where:
- w = portfolio weights
- μ = expected returns (from Elastic Net model)
- Σ = covariance matrix (using GARCH volatilities)
- λ = risk aversion parameter
- rf = risk-free rate

Enhancement with GARCH:
----------------------
Traditional MVO uses historical covariance, which:
1. Assumes constant volatility (violated in practice)
2. Gives equal weight to old and recent observations
3. Fails to capture volatility clustering

Our approach:
1. Forecast volatility for each stock using GARCH(1,1)
2. Use DCC-like construction: Σ = D·R·D where D = diag(GARCH vols)
3. Apply Ledoit-Wolf shrinkage to the correlation matrix R

References:
----------
- Markowitz, H. (1952). "Portfolio Selection." Journal of Finance, 7(1), 77-91.
- Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional 
  Covariance Matrices." Journal of Multivariate Analysis, 88(2), 365-411.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import (
    PORTFOLIO_SIZE, LONG_ONLY, EQUAL_WEIGHT, OPTIMIZATION_METHOD,
    MAX_POSITION_SIZE, MAX_SECTOR_EXPOSURE, MAX_COUNTRY_EXPOSURE
)
from ..models.garch_model import PortfolioGARCH, GARCHModel


class PortfolioOptimizer:
    """
    GARCH-enhanced Mean-Variance Portfolio Optimizer.
    
    This optimizer:
    1. Filters stocks to those with positive expected returns
    2. Forecasts volatility using GARCH(1,1) for each stock
    3. Constructs covariance matrix with GARCH vols and shrunk correlations
    4. Maximizes Sharpe Ratio subject to constraints
    5. Lets the optimization determine effective portfolio size
    """
    
    def __init__(self, 
                 portfolio_size: int = PORTFOLIO_SIZE,
                 equal_weight: bool = EQUAL_WEIGHT,
                 max_position: float = MAX_POSITION_SIZE,
                 min_position: float = 0.01,
                 optimization_method: str = OPTIMIZATION_METHOD,
                 risk_free_rate: float = 0.03):
        """
        Initialize optimizer.
        
        Args:
            portfolio_size: Maximum number of stocks (model may choose fewer)
            equal_weight: If True, use equal weighting (simple approach)
            max_position: Maximum weight per stock (e.g., 0.10 = 10%)
            min_position: Minimum weight to be considered in portfolio
            optimization_method: 'garch_mv', 'inverse_vol', 'risk_parity', 'max_sharpe'
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.portfolio_size = portfolio_size
        self.equal_weight = equal_weight
        self.max_position = max_position
        self.min_position = min_position
        self.optimization_method = optimization_method
        self.risk_free_rate = risk_free_rate
        self.long_only = LONG_ONLY
        
        # Store optimization details for dashboard
        self.garch_models = None
        self.covariance_matrix = None
        self.optimization_result = None
        
    def construct_portfolio(self, 
                           expected_returns: pd.Series,
                           stock_info: Optional[pd.DataFrame] = None,
                           historical_returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Construct optimized portfolio using GARCH-enhanced Mean-Variance optimization.
        
        The model decides:
        1. Which stocks to include (positive expected return filter)
        2. How many stocks (optimization may zero-out some)
        3. What weights (based on risk-return tradeoff)
        
        Args:
            expected_returns: Series with ticker -> model-implied expected return
            stock_info: DataFrame with stock metadata (sector, country, etc.)
            historical_returns: DataFrame of historical returns for covariance estimation
            
        Returns:
            DataFrame with columns: ticker, weight, expected_return, garch_vol, etc.
        """
        print("\n" + "=" * 60)
        print("PORTFOLIO OPTIMIZATION")
        print("=" * 60)
        
        # Step 1: Filter to stocks with positive expected returns
        print("\n[1/4] Filtering stocks with positive expected returns...")
        positive_returns = expected_returns[expected_returns > 0]
        print(f"      {len(positive_returns)}/{len(expected_returns)} stocks have positive expected returns")
        
        if len(positive_returns) == 0:
            print("      Warning: No stocks with positive expected return. Using top stocks.")
            positive_returns = expected_returns.nlargest(self.portfolio_size)
        
        # Limit to top N candidates for computational efficiency
        candidates = positive_returns.nlargest(min(self.portfolio_size * 2, len(positive_returns)))
        
        if self.equal_weight:
            # Simple equal weighting (not recommended but available)
            print("\n[2/4] Using equal-weight allocation (EQUAL_WEIGHT=True)")
            top_stocks = candidates.nlargest(self.portfolio_size)
            weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
            garch_vols = pd.Series(np.nan, index=top_stocks.index)
        else:
            # GARCH-enhanced optimization
            print(f"\n[2/4] Running {self.optimization_method} optimization...")
            
            if historical_returns is None:
                print("      Warning: No historical returns provided. Using equal weights.")
                top_stocks = candidates.nlargest(self.portfolio_size)
                weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
                garch_vols = pd.Series(np.nan, index=top_stocks.index)
            else:
                # Get returns for candidate stocks only
                available_stocks = [s for s in candidates.index if s in historical_returns.columns]
                if len(available_stocks) < 5:
                    print("      Warning: Insufficient historical data. Using equal weights.")
                    top_stocks = candidates.nlargest(self.portfolio_size)
                    weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
                    garch_vols = pd.Series(np.nan, index=top_stocks.index)
                else:
                    returns_subset = historical_returns[available_stocks]
                    expected_subset = candidates.reindex(available_stocks).dropna()
                    
                    weights, garch_vols = self._optimize_with_garch(
                        expected_subset, 
                        returns_subset
                    )
        
        # Step 3: Filter out zero weights
        print("\n[3/4] Filtering portfolio to non-zero weights...")
        non_zero = weights[weights >= self.min_position]
        if len(non_zero) == 0:
            print("      Warning: All weights below threshold. Using equal weights.")
            top_stocks = candidates.nlargest(min(self.portfolio_size, len(candidates)))
            weights = pd.Series(1.0 / len(top_stocks), index=top_stocks.index)
        else:
            weights = non_zero / non_zero.sum()  # Renormalize
        
        print(f"      Final portfolio: {len(weights)} positions")
        
        # Step 4: Construct portfolio DataFrame
        print("\n[4/4] Constructing portfolio DataFrame...")
        portfolio = pd.DataFrame({
            'ticker': weights.index,
            'weight': weights.values,
            'expected_return': expected_returns.reindex(weights.index).values,
            'garch_volatility': garch_vols.reindex(weights.index).values if garch_vols is not None else np.nan
        })
        
        # Sort by weight
        portfolio = portfolio.sort_values('weight', ascending=False)
        
        # Add metadata if available
        if stock_info is not None and len(stock_info) > 0:
            portfolio = self._add_stock_metadata(portfolio, stock_info)
        
        print(f"\n      Portfolio expected return: {(portfolio['weight'] * portfolio['expected_return']).sum():.4f}")
        print(f"      Top position: {portfolio['ticker'].iloc[0]} ({portfolio['weight'].iloc[0]:.1%})")
        
        return portfolio
    
    def _optimize_with_garch(self, 
                             expected_returns: pd.Series,
                             historical_returns: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Perform GARCH-enhanced mean-variance optimization.
        
        Args:
            expected_returns: Expected returns for candidate stocks
            historical_returns: Historical returns for GARCH and correlation estimation
            
        Returns:
            Tuple of (weights, garch_volatilities)
        """
        stocks = expected_returns.index.tolist()
        n = len(stocks)
        
        print(f"      Fitting GARCH(1,1) models for {n} stocks...")
        
        # Fit GARCH models
        self.garch_models = PortfolioGARCH()
        self.garch_models.fit_all(historical_returns[stocks], verbose=False)
        
        # Get volatility forecasts
        garch_vols = self.garch_models.forecast_volatilities(horizon=1)
        
        # Fill NaN volatilities with historical
        hist_vols = historical_returns[stocks].std() * np.sqrt(252)  # Annualize
        garch_vols = garch_vols.fillna(hist_vols)
        
        # Get covariance matrix
        print("      Constructing covariance matrix with GARCH volatilities...")
        self.covariance_matrix = self.garch_models.get_covariance_matrix(
            historical_returns[stocks], 
            use_shrinkage=True
        )
        
        # Annualize covariance (assuming daily returns)
        cov_annual = self.covariance_matrix * 252
        
        # Convert expected returns to annual (they're daily from Elastic Net)
        mu = expected_returns * 252
        
        # Run optimization
        if self.optimization_method in ['garch_mv', 'max_sharpe']:
            weights = self._maximize_sharpe(mu, cov_annual)
        elif self.optimization_method == 'inverse_vol':
            weights = self._inverse_volatility(garch_vols)
        elif self.optimization_method == 'risk_parity':
            weights = self._risk_parity(cov_annual)
        else:
            weights = self._maximize_sharpe(mu, cov_annual)
        
        return weights, garch_vols * np.sqrt(252)  # Return annualized vols
    
    def _maximize_sharpe(self, 
                        expected_returns: pd.Series, 
                        covariance: pd.DataFrame) -> pd.Series:
        """
        Maximize Sharpe Ratio subject to constraints.
        
        Solves:
            max  (w'μ - rf) / √(w'Σw)
            s.t. w'1 = 1
                 0 ≤ w_i ≤ max_position
        """
        n = len(expected_returns)
        mu = expected_returns.values
        Sigma = covariance.values
        
        # Ensure positive definiteness
        min_eig = np.min(np.linalg.eigvalsh(Sigma))
        if min_eig < 0:
            Sigma = Sigma + (-min_eig + 1e-6) * np.eye(n)
        
        def neg_sharpe(w):
            port_return = np.dot(w, mu)
            port_vol = np.sqrt(np.dot(w, np.dot(Sigma, w)))
            if port_vol < 1e-10:
                return 1e10
            return -(port_return - self.risk_free_rate) / port_vol
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds (long-only with max position)
        if self.long_only:
            bounds = [(0, self.max_position) for _ in range(n)]
        else:
            bounds = [(-self.max_position, self.max_position) for _ in range(n)]
        
        # Initial guess (equal weight)
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            neg_sharpe, 
            w0, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        self.optimization_result = {
            'success': result.success,
            'sharpe_ratio': -result.fun,
            'message': result.message
        }
        
        if result.success:
            print(f"      Optimization successful! Sharpe Ratio: {-result.fun:.3f}")
        else:
            print(f"      Optimization warning: {result.message}")
        
        weights = pd.Series(result.x, index=expected_returns.index)
        return weights
    
    def _inverse_volatility(self, volatilities: pd.Series) -> pd.Series:
        """
        Inverse volatility weighting: allocate more to less volatile stocks.
        
        w_i = (1/σ_i) / Σ(1/σ_j)
        """
        inv_vol = 1 / volatilities.replace(0, np.nan).fillna(volatilities.mean())
        weights = inv_vol / inv_vol.sum()
        
        # Apply max position constraint
        weights = weights.clip(upper=self.max_position)
        weights = weights / weights.sum()
        
        return weights
    
    def _risk_parity(self, covariance: pd.DataFrame) -> pd.Series:
        """
        Risk parity: equal risk contribution from each asset.
        
        Each asset contributes equally to total portfolio variance.
        """
        n = len(covariance)
        Sigma = covariance.values
        
        def risk_contribution_error(w):
            w = np.array(w)
            port_var = np.dot(w, np.dot(Sigma, w))
            marginal_contrib = np.dot(Sigma, w)
            risk_contrib = w * marginal_contrib / np.sqrt(port_var)
            target_contrib = np.sqrt(port_var) / n
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0.01, self.max_position) for _ in range(n)]
        w0 = np.ones(n) / n
        
        result = minimize(
            risk_contribution_error,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights = pd.Series(result.x, index=covariance.index)
        return weights
    
    def _add_stock_metadata(self, portfolio: pd.DataFrame, stock_info: pd.DataFrame) -> pd.DataFrame:
        """Add stock metadata to portfolio."""
        try:
            info_df = stock_info.reset_index()
            
            ticker_col = None
            for col in ['ticker', 'index', info_df.columns[0]]:
                if col in info_df.columns:
                    ticker_col = col
                    break
            
            if ticker_col and ticker_col != 'ticker':
                info_df = info_df.rename(columns={ticker_col: 'ticker'})
            
            if 'ticker' in info_df.columns:
                portfolio = portfolio.merge(info_df, on='ticker', how='left')
        except Exception:
            pass
        
        return portfolio
    
    def calculate_portfolio_metrics(self, portfolio: pd.DataFrame, 
                                   historical_returns: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate comprehensive portfolio metrics.
        
        Args:
            portfolio: Portfolio DataFrame with weights and expected returns
            historical_returns: Historical returns for realized metrics
            
        Returns:
            Dictionary of metrics with academic interpretations
        """
        weights = portfolio['weight'].values
        exp_returns = portfolio['expected_return'].values
        
        # Expected return (annualized)
        portfolio_return = np.dot(weights, exp_returns) * 252
        
        # Concentration metrics
        herfindahl = np.sum(weights ** 2)
        effective_n = 1 / herfindahl
        
        metrics = {
            'expected_return_annual': portfolio_return,
            'n_positions': len(portfolio),
            'concentration_herfindahl': herfindahl,
            'effective_n_positions': effective_n,
            'max_position': weights.max(),
            'min_position': weights[weights > 0].min() if (weights > 0).any() else 0,
        }
        
        # If we have covariance matrix, calculate portfolio volatility
        if self.covariance_matrix is not None:
            try:
                port_stocks = portfolio['ticker'].tolist()
                cov_subset = self.covariance_matrix.reindex(
                    index=port_stocks, columns=port_stocks
                ).fillna(0)
                port_var = np.dot(weights, np.dot(cov_subset.values, weights))
                port_vol = np.sqrt(port_var) * np.sqrt(252)  # Annualize
                
                metrics['expected_volatility_annual'] = port_vol
                metrics['expected_sharpe_ratio'] = (portfolio_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
            except Exception:
                pass
        
        # GARCH model summary
        if self.garch_models is not None:
            try:
                garch_summary = self.garch_models.get_model_summary()
                avg_persistence = garch_summary['persistence'].mean()
                metrics['avg_garch_persistence'] = avg_persistence
                metrics['garch_interpretation'] = (
                    "High volatility persistence" if avg_persistence > 0.9 
                    else "Moderate volatility persistence" if avg_persistence > 0.7 
                    else "Low volatility persistence"
                )
            except Exception:
                pass
        
        # Sector/country breakdown
        if 'sector' in portfolio.columns:
            sector_weights = portfolio.groupby('sector')['weight'].sum()
            metrics['sector_breakdown'] = sector_weights.to_dict()
            metrics['max_sector_exposure'] = sector_weights.max()
        
        if 'country' in portfolio.columns:
            country_weights = portfolio.groupby('country')['weight'].sum()
            metrics['country_breakdown'] = country_weights.to_dict()
            metrics['max_country_exposure'] = country_weights.max()
        
        return metrics


# Academic explanation for dashboard
OPTIMIZATION_EXPLANATION = """
## Mean-Variance Optimization with GARCH - Academic Background

### Modern Portfolio Theory (MPT)

Harry Markowitz's (1952) Nobel Prize-winning framework establishes that rational investors 
should construct portfolios that maximize expected return for a given level of risk:

$$\\max_w \\quad w'\\mu - \\frac{\\lambda}{2} w'\\Sigma w$$

Subject to: $w'\\mathbf{1} = 1$, $w \\geq 0$ (long-only)

Where:
- $w$ = vector of portfolio weights
- $\\mu$ = vector of expected returns (from our Elastic Net model)
- $\\Sigma$ = covariance matrix (enhanced with GARCH)
- $\\lambda$ = risk aversion parameter

### Maximum Sharpe Ratio

Equivalently, we maximize the **Sharpe Ratio** (risk-adjusted return):

$$\\max_w \\frac{w'\\mu - r_f}{\\sqrt{w'\\Sigma w}}$$

This finds the portfolio on the efficient frontier with the steepest risk-return tradeoff.

### GARCH Enhancement

Traditional MVO uses historical covariance, which has several problems:
1. **Constant volatility assumption** - violated in practice
2. **Equal weighting of observations** - old data weighted same as recent
3. **No volatility clustering** - fails to capture market dynamics

Our approach constructs the covariance matrix as:

$$\\Sigma = D \\cdot R \\cdot D$$

Where:
- $D$ = diagonal matrix of GARCH(1,1) volatility forecasts
- $R$ = correlation matrix (with Ledoit-Wolf shrinkage)

### Ledoit-Wolf Shrinkage

For large portfolios, sample correlation matrices are noisy. We apply shrinkage:

$$R_{shrunk} = (1-\\delta) R_{sample} + \\delta R_{target}$$

Where $\\delta$ is the shrinkage intensity and $R_{target}$ is a structured target 
(e.g., constant correlation or identity matrix).

### Constraints

Our optimization includes practical constraints:
- **Long-only**: $w_i \\geq 0$ (no short selling)
- **Maximum position**: $w_i \\leq 10\\%$ (diversification)
- **Full investment**: $\\sum w_i = 1$

### References

- Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance*, 7(1), 77-91.
- Sharpe, W.F. (1966). "Mutual Fund Performance." *Journal of Business*, 39(1), 119-138.
- Ledoit, O., & Wolf, M. (2004). "A Well-Conditioned Estimator for Large-Dimensional 
  Covariance Matrices." *Journal of Multivariate Analysis*, 88(2), 365-411.
"""


if __name__ == "__main__":
    print("Portfolio optimization module loaded")
    print(f"Optimization method: {OPTIMIZATION_METHOD}")
    print(f"Equal weight: {EQUAL_WEIGHT}")