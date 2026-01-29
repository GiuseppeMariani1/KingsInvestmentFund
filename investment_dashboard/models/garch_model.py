"""
GARCH(1,1) Volatility Forecasting Model

Academic Background:
====================
The Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model,
introduced by Bollerslev (1986), extends Engle's (1982) ARCH model to capture
the time-varying nature of financial return volatility.

GARCH(1,1) Specification:
-------------------------
The GARCH(1,1) model specifies conditional variance as:

    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Where:
- σ²_t: Conditional variance at time t
- ω (omega): Long-run average variance weight (ω > 0)
- α (alpha): ARCH coefficient - impact of past shocks (α ≥ 0)
- β (beta): GARCH coefficient - persistence of volatility (β ≥ 0)
- ε_{t-1}: Previous period's residual (innovation)
- Stationarity requires: α + β < 1

Interpretation:
--------------
- High α: Volatility reacts strongly to market movements (nervous market)
- High β: Volatility is persistent (volatility clustering)
- α + β close to 1: High persistence, shocks decay slowly
- Unconditional variance: σ² = ω / (1 - α - β)

References:
----------
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity."
  Journal of Econometrics, 31(3), 307-327.
- Engle, R.F. (1982). "Autoregressive Conditional Heteroscedasticity with Estimates
  of the Variance of United Kingdom Inflation." Econometrica, 50(4), 987-1007.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Try to import arch library, fall back to manual implementation if not available
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: 'arch' library not available. Using manual GARCH implementation.")


class GARCHModel:
    """
    GARCH(1,1) model for volatility forecasting.
    
    This implementation supports both the 'arch' library (if available) and
    a manual maximum likelihood estimation fallback.
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Initialize GARCH(p,q) model.
        
        Args:
            p: Order of GARCH terms (lagged conditional variances)
            q: Order of ARCH terms (lagged squared residuals)
        """
        self.p = p
        self.q = q
        self.params = None
        self.omega = None
        self.alpha = None
        self.beta = None
        self.conditional_volatility = None
        self.model_fit = None
        
    def fit(self, returns: np.ndarray, verbose: bool = False) -> 'GARCHModel':
        """
        Fit GARCH(1,1) model to return series.
        
        Args:
            returns: Array of returns (should be in percentage or decimal form)
            verbose: Print fitting details
            
        Returns:
            Self with fitted parameters
        """
        # Clean returns
        returns = np.asarray(returns).flatten()
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 30:
            raise ValueError("Insufficient data for GARCH estimation (need at least 30 observations)")
        
        # Scale returns to percentage if they appear to be in decimal form
        if np.abs(returns).mean() < 0.1:
            returns = returns * 100
        
        if ARCH_AVAILABLE:
            self._fit_with_arch(returns, verbose)
        else:
            self._fit_manual(returns, verbose)
            
        return self
    
    def _fit_with_arch(self, returns: np.ndarray, verbose: bool):
        """Fit using the arch library."""
        model = arch_model(returns, vol='Garch', p=self.p, q=self.q, 
                          mean='Constant', rescale=True)
        self.model_fit = model.fit(disp='off' if not verbose else 'on')
        
        # Extract parameters
        self.omega = self.model_fit.params['omega']
        self.alpha = self.model_fit.params['alpha[1]']
        self.beta = self.model_fit.params['beta[1]']
        self.params = {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'persistence': self.alpha + self.beta,
            'unconditional_vol': np.sqrt(self.omega / (1 - self.alpha - self.beta)) if (self.alpha + self.beta) < 1 else np.nan
        }
        self.conditional_volatility = self.model_fit.conditional_volatility
        
    def _fit_manual(self, returns: np.ndarray, verbose: bool):
        """Manual GARCH(1,1) estimation via Maximum Likelihood."""
        T = len(returns)
        
        # Initial parameter guesses
        var_returns = np.var(returns)
        initial_params = [var_returns * 0.1, 0.1, 0.8]  # omega, alpha, beta
        
        # Bounds: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        bounds = [(1e-6, var_returns * 10), (1e-6, 0.5), (1e-6, 0.999)]
        
        def neg_log_likelihood(params):
            omega, alpha, beta = params
            if alpha + beta >= 1:
                return 1e10
            
            sigma2 = np.zeros(T)
            sigma2[0] = var_returns
            
            for t in range(1, T):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            
            # Log-likelihood (ignoring constant)
            ll = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
            return -ll
        
        result = minimize(neg_log_likelihood, initial_params, method='L-BFGS-B', bounds=bounds)
        
        self.omega, self.alpha, self.beta = result.x
        self.params = {
            'omega': self.omega,
            'alpha': self.alpha,
            'beta': self.beta,
            'persistence': self.alpha + self.beta,
            'unconditional_vol': np.sqrt(self.omega / (1 - self.alpha - self.beta)) if (self.alpha + self.beta) < 1 else np.nan
        }
        
        # Calculate conditional volatility
        sigma2 = np.zeros(T)
        sigma2[0] = var_returns
        for t in range(1, T):
            sigma2[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * sigma2[t-1]
        self.conditional_volatility = np.sqrt(sigma2)
        
    def forecast(self, horizon: int = 1) -> np.ndarray:
        """
        Forecast volatility for future periods.
        
        Args:
            horizon: Number of periods to forecast
            
        Returns:
            Array of forecasted volatilities
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if ARCH_AVAILABLE and self.model_fit is not None:
            forecast = self.model_fit.forecast(horizon=horizon)
            return np.sqrt(forecast.variance.values[-1, :])
        else:
            # Manual forecast
            forecasts = np.zeros(horizon)
            last_var = self.conditional_volatility[-1]**2
            
            for h in range(horizon):
                if h == 0:
                    forecasts[h] = self.omega + (self.alpha + self.beta) * last_var
                else:
                    forecasts[h] = self.omega + (self.alpha + self.beta) * forecasts[h-1]
            
            return np.sqrt(forecasts)
    
    def get_summary(self) -> Dict:
        """Get model summary with academic interpretation."""
        if self.params is None:
            return {}
        
        persistence = self.alpha + self.beta
        half_life = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf
        
        return {
            'parameters': self.params,
            'interpretation': {
                'volatility_persistence': f"{'High' if persistence > 0.9 else 'Moderate' if persistence > 0.7 else 'Low'} ({persistence:.3f})",
                'shock_impact': f"{'Strong' if self.alpha > 0.15 else 'Moderate' if self.alpha > 0.08 else 'Weak'} (α={self.alpha:.3f})",
                'half_life_days': f"{half_life:.1f} days" if half_life < 1000 else "Very persistent",
                'unconditional_vol_annual': f"{self.params['unconditional_vol'] * np.sqrt(252):.1f}%" if not np.isnan(self.params['unconditional_vol']) else "N/A"
            }
        }


class PortfolioGARCH:
    """
    GARCH-based portfolio volatility estimation and forecasting.
    
    This class fits GARCH(1,1) models to each asset and constructs
    a covariance matrix for portfolio optimization.
    """
    
    def __init__(self):
        self.models = {}
        self.volatility_forecasts = {}
        self.correlation_matrix = None
        self.covariance_matrix = None
        
    def fit_all(self, returns: pd.DataFrame, verbose: bool = False) -> 'PortfolioGARCH':
        """
        Fit GARCH(1,1) to all assets.
        
        Args:
            returns: DataFrame with assets as columns, dates as index
            verbose: Print progress
            
        Returns:
            Self with fitted models
        """
        if verbose:
            print(f"Fitting GARCH(1,1) to {len(returns.columns)} assets...")
        
        successful = 0
        failed = []
        
        for i, col in enumerate(returns.columns):
            try:
                model = GARCHModel(p=1, q=1)
                model.fit(returns[col].dropna().values, verbose=False)
                self.models[col] = model
                successful += 1
            except Exception as e:
                failed.append(col)
                # Use simple historical volatility as fallback
                self.models[col] = None
                
            if verbose and (i + 1) % 50 == 0:
                print(f"  Fitted {i + 1}/{len(returns.columns)} models...")
        
        if verbose:
            print(f"  GARCH fitting complete: {successful} successful, {len(failed)} failed")
        
        # Calculate correlation matrix from standardized returns
        self.correlation_matrix = returns.corr()
        
        return self
    
    def forecast_volatilities(self, horizon: int = 1) -> pd.Series:
        """
        Forecast volatility for all assets.
        
        Args:
            horizon: Forecast horizon in days
            
        Returns:
            Series of forecasted volatilities
        """
        forecasts = {}
        
        for asset, model in self.models.items():
            if model is not None and model.params is not None:
                forecasts[asset] = model.forecast(horizon)[0]
            else:
                # Fallback: use last conditional volatility or NaN
                forecasts[asset] = np.nan
        
        self.volatility_forecasts = pd.Series(forecasts)
        return self.volatility_forecasts
    
    def get_covariance_matrix(self, returns: pd.DataFrame, use_shrinkage: bool = True) -> pd.DataFrame:
        """
        Construct covariance matrix using GARCH volatilities and correlations.
        
        Uses the DCC-like approach:
        Σ = D · R · D
        
        Where:
        - D = diagonal matrix of GARCH volatilities
        - R = correlation matrix (with optional shrinkage)
        
        Args:
            returns: Historical returns for correlation estimation
            use_shrinkage: Apply Ledoit-Wolf shrinkage to correlation matrix
            
        Returns:
            Covariance matrix as DataFrame
        """
        # Get GARCH volatility forecasts
        if not self.volatility_forecasts.any():
            self.forecast_volatilities(horizon=1)
        
        assets = returns.columns
        n = len(assets)
        
        # Get volatilities (fill NaN with historical vol)
        hist_vol = returns.std()
        vols = self.volatility_forecasts.reindex(assets)
        vols = vols.fillna(hist_vol)
        
        # Get correlation matrix
        if use_shrinkage:
            corr = self._shrink_correlation(returns)
        else:
            corr = returns.corr()
        
        # Construct covariance: Σ = D · R · D
        D = np.diag(vols.values)
        cov_matrix = D @ corr.values @ D
        
        self.covariance_matrix = pd.DataFrame(cov_matrix, index=assets, columns=assets)
        return self.covariance_matrix
    
    def _shrink_correlation(self, returns: pd.DataFrame, shrinkage_target: str = 'constant') -> pd.DataFrame:
        """
        Apply Ledoit-Wolf shrinkage to correlation matrix.
        
        Shrinkage helps stabilize the correlation matrix estimation,
        especially important when n (assets) is large relative to T (observations).
        
        Reference: Ledoit & Wolf (2004) "A Well-Conditioned Estimator for 
        Large-Dimensional Covariance Matrices"
        """
        sample_corr = returns.corr()
        n = len(sample_corr)
        
        if shrinkage_target == 'constant':
            # Shrink towards constant correlation
            avg_corr = (sample_corr.values.sum() - n) / (n * (n - 1))
            target = np.full((n, n), avg_corr)
            np.fill_diagonal(target, 1.0)
        else:
            # Shrink towards identity
            target = np.eye(n)
        
        # Simple shrinkage intensity (can be optimized)
        T = len(returns)
        shrinkage_intensity = min(1.0, max(0.0, (n / T) * 0.5))
        
        shrunk_corr = (1 - shrinkage_intensity) * sample_corr.values + shrinkage_intensity * target
        
        return pd.DataFrame(shrunk_corr, index=sample_corr.index, columns=sample_corr.columns)
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all GARCH models."""
        summaries = []
        
        for asset, model in self.models.items():
            if model is not None and model.params is not None:
                summaries.append({
                    'asset': asset,
                    'omega': model.omega,
                    'alpha': model.alpha,
                    'beta': model.beta,
                    'persistence': model.alpha + model.beta,
                    'unconditional_vol': model.params['unconditional_vol']
                })
        
        return pd.DataFrame(summaries)


# Academic explanation for dashboard
GARCH_EXPLANATION = """
## GARCH(1,1) Model - Academic Background

### What is GARCH?
**Generalized Autoregressive Conditional Heteroskedasticity (GARCH)** is a statistical model 
that captures the time-varying nature of financial market volatility. Introduced by Tim Bollerslev 
in 1986, it extends Robert Engle's 1982 ARCH model (which won the Nobel Prize in Economics in 2003).

### The GARCH(1,1) Specification

The conditional variance evolves according to:

$$\\sigma^2_t = \\omega + \\alpha \\varepsilon^2_{t-1} + \\beta \\sigma^2_{t-1}$$

Where:
- $\\sigma^2_t$ = Conditional variance at time t (what we're forecasting)
- $\\omega$ = Long-run variance weight (baseline volatility)
- $\\alpha$ = ARCH coefficient (how much past shocks affect current volatility)
- $\\beta$ = GARCH coefficient (how persistent volatility is)
- $\\varepsilon_{t-1}$ = Previous period's return shock

### Key Properties

1. **Volatility Clustering**: GARCH captures the empirical fact that large returns tend to be 
   followed by large returns (of either sign), and small returns by small returns.

2. **Mean Reversion**: Volatility eventually reverts to its long-run level:
   $$\\bar{\\sigma}^2 = \\frac{\\omega}{1 - \\alpha - \\beta}$$

3. **Persistence**: The sum $\\alpha + \\beta$ measures how slowly volatility shocks decay.
   - Close to 1 = highly persistent (shocks last a long time)
   - Close to 0 = mean-reverting quickly

### Why Use GARCH for Portfolio Optimization?

Traditional mean-variance optimization uses historical volatility, which assumes constant risk.
GARCH provides **forward-looking volatility estimates** that:

1. React to recent market conditions
2. Capture volatility clustering
3. Provide more accurate risk forecasts during turbulent periods
4. Lead to better risk-adjusted portfolio weights

### References
- Bollerslev, T. (1986). "Generalized Autoregressive Conditional Heteroskedasticity." 
  *Journal of Econometrics*, 31(3), 307-327.
- Engle, R.F. (2001). "GARCH 101: The Use of ARCH/GARCH Models in Applied Econometrics." 
  *Journal of Economic Perspectives*, 15(4), 157-168.
"""


if __name__ == "__main__":
    # Test GARCH model
    np.random.seed(42)
    
    # Generate synthetic GARCH data
    T = 500
    omega, alpha, beta = 0.1, 0.1, 0.85
    
    returns = np.zeros(T)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)
    
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.sqrt(sigma2[t]) * np.random.randn()
    
    # Fit model
    model = GARCHModel()
    model.fit(returns, verbose=True)
    
    print("\nTrue parameters: omega=0.1, alpha=0.1, beta=0.85")
    print(f"Estimated: omega={model.omega:.4f}, alpha={model.alpha:.4f}, beta={model.beta:.4f}")
    print(f"\nModel Summary:")
    print(model.get_summary())
