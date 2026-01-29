"""
Ridge Regression Model

For each stock, evaluates how strongly and consistently it benefits from the PLS components
(return drivers). Ridge regression shrinks all coefficients proportionally, providing stable
estimates even when features are correlated.

Strategy: Ridge regression identifies stocks with:
- Strong exposure to return-driving components
- Consistent relationships (not spurious correlations)
- Robust signals (survives cross-validation)

Why Ridge (not Lasso or Elastic Net):
- PLS already selected features (the components ARE the return drivers)
- We want to keep ALL component exposures, not zero any out
- Ridge provides stable estimates when features are correlated
- Simpler model with one hyperparameter (alpha only)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import ELASTIC_NET_LOOKBACK_YEARS


class ElasticNetModel:
    """
    Ridge regression to estimate expected returns at stock level.
    Uses PLS components as features and applies L2 regularization.
    
    Note: Class name kept as ElasticNetModel for backwards compatibility,
    but internally uses pure Ridge regression.
    
    Alpha (regularization strength) is selected via cross-validation.
    """
    
    def __init__(self, l1_ratios: List[float] = None, cv_folds: int = 10):
        """
        Initialize Ridge model.
        
        Args:
            l1_ratios: Ignored (kept for backwards compatibility)
            cv_folds: Number of CV folds for alpha selection
        """
        self.cv_folds = cv_folds
        
        # Alpha values to test (regularization strength)
        # Lower = less regularization = more signal spread
        self.alphas = np.logspace(-4, 2, 50)  # 0.0001 to 100
        
        self.models = {}  # One model per stock
        self.scalers = {}  # One scaler per stock
        self.expected_returns = None
        self.coefficients = None
        self.fit_metrics = {}
        
        # CV results storage for visualization
        self.cv_results = {
            'alpha_distribution': [],     # Distribution of selected alphas
            'l1_ratio_distribution': [],  # Always 0 for Ridge (kept for compatibility)
            'cv_errors_by_alpha': {},
            'aggregate_cv_curve': None
        }
        
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, 
            stock_characteristics: pd.DataFrame = None) -> 'ElasticNetModel':
        """
        Fit Ridge models for each stock with CV for alpha selection.
        
        For each stock, estimates sensitivity (beta) to PLS components.
        Alpha is selected via time-series cross-validation.
        
        Args:
            X: PLS component scores (dates x components) - the return-driving factors
            Y: Stock returns (dates x stocks) - individual stock returns
            stock_characteristics: Stock-level factors (not used, kept for compatibility)
            
        Returns:
            self
        """
        print(f"      Fitting Ridge models with {self.cv_folds}-fold CV...")
        print(f"      Testing {len(self.alphas)} alpha values")
        
        # Store stock characteristics for later use
        self.stock_characteristics = stock_characteristics
        
        # Align data
        common_dates = X.index.intersection(Y.index)
        X_aligned = X.loc[common_dates].copy()
        Y_aligned = Y.loc[common_dates].copy()
        
        # Handle NaN and inf values
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        Y_aligned = Y_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if len(common_dates) < 50:
            raise ValueError(f"Insufficient data: only {len(common_dates)} common dates")
        
        # Time series cross-validation (respects temporal ordering)
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Reset CV results
        self.cv_results = {
            'alpha_distribution': [],
            'l1_ratio_distribution': [],  # Will be all zeros for Ridge
            'cv_errors_by_alpha': {},
            'aggregate_cv_curve': None
        }
        
        # Fit model for each stock
        coefficients_list = []
        expected_returns_list = []
        n_stocks = len(Y_aligned.columns)
        
        # Track CV errors for aggregate curve
        all_cv_scores = []
        
        for i, stock in enumerate(Y_aligned.columns):
            if (i + 1) % 20 == 0:
                print(f"      Progress: {i+1}/{n_stocks} stocks...")
            
            y_stock = Y_aligned[stock].copy()
            X_stock = X_aligned.copy()
            
            # Clean this stock's data
            y_stock = y_stock.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if len(y_stock) < 30:
                continue
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_stock.values)
            y_values = y_stock.values
            
            # Final NaN check
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            y_values = np.nan_to_num(y_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ridge CV: finds optimal alpha
            model = RidgeCV(
                alphas=self.alphas,
                cv=tscv,
                scoring='neg_mean_squared_error'
            )
            
            try:
                model.fit(X_scaled, y_values)
                
                # Store model and scaler
                self.models[stock] = model
                self.scalers[stock] = scaler
                
                # Record selected hyperparameters
                self.cv_results['alpha_distribution'].append(model.alpha_)
                self.cv_results['l1_ratio_distribution'].append(0.0)  # Ridge = l1_ratio of 0
                
                # Calculate expected return (using latest data)
                latest_X = X_scaled[-1:].reshape(1, -1)
                expected_return = model.predict(latest_X)[0]
                expected_returns_list.append({
                    'ticker': stock,
                    'expected_return': expected_return
                })
                
                # Store coefficients
                coefs = pd.Series(model.coef_, index=X_stock.columns, name=stock)
                coefficients_list.append(coefs)
                
                # Store fit metrics
                r2 = model.score(X_scaled, y_values)
                n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)  # All should be non-zero for Ridge
                self.fit_metrics[stock] = {
                    'r2': r2,
                    'alpha': model.alpha_,
                    'l1_ratio': 0.0,  # Ridge
                    'n_features': n_nonzero
                }
                
            except Exception as e:
                # Silently skip problematic stocks
                continue
        
        # Compute aggregate statistics
        self.cv_results['aggregate_cv_curve'] = {
            'l1_ratios': [0.0],  # Only Ridge
            'mean_cv_error': [0.0],  # Placeholder
            'std_cv_error': [0.0]
        }
        
        # Combine results
        self.expected_returns = pd.DataFrame(expected_returns_list).set_index('ticker')
        self.coefficients = pd.DataFrame(coefficients_list).T
        
        avg_r2 = np.mean([m['r2'] for m in self.fit_metrics.values()]) if self.fit_metrics else 0
        avg_alpha = np.mean(self.cv_results['alpha_distribution']) if self.cv_results['alpha_distribution'] else 0
        
        print(f"      Fitted models for {len(self.models)} stocks")
        print(f"      Average R-squared: {avg_r2:.4f}")
        print(f"      Average alpha (CV-selected): {avg_alpha:.4f}")
        print(f"      Model type: Pure Ridge (l1_ratio = 0)")
        
        return self
    
    def get_cv_results(self) -> Dict:
        """Get cross-validation results for visualization"""
        return self.cv_results
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict expected returns for new PLS component data
        
        Args:
            X: PLS component scores (dates x components)
            
        Returns:
            Expected returns (stocks x expected_return)
        """
        if not self.models:
            raise ValueError("Models not fitted. Call fit() first.")
        
        predictions = []
        
        for stock, model in self.models.items():
            scaler = self.scalers[stock]
            X_scaled = scaler.transform(X.values)
            pred = model.predict(X_scaled)
            predictions.append({
                'ticker': stock,
                'expected_return': pred[-1] if len(pred) > 0 else np.nan
            })
        
        return pd.DataFrame(predictions).set_index('ticker')
    
    def get_expected_returns(self) -> pd.Series:
        """Get expected returns for all stocks"""
        if self.expected_returns is None:
            raise ValueError("Model not fitted.")
        return self.expected_returns['expected_return']
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get model coefficients (exposures to PLS components)"""
        return self.coefficients.copy()
    
    def get_fit_metrics(self) -> pd.DataFrame:
        """Get model fit metrics"""
        return pd.DataFrame(self.fit_metrics).T
    
    def get_stock_exposures(self, ticker: str) -> pd.Series:
        """Get a stock's exposure to each PLS component"""
        if ticker not in self.coefficients.columns:
            raise ValueError(f"Stock {ticker} not in model")
        return self.coefficients[ticker]


if __name__ == "__main__":
    # Test Ridge model
    print("Ridge regression model module loaded")
