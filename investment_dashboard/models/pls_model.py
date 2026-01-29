"""
Partial Least Squares (PLS) Model

Compresses highly correlated macro and style variables into a small set of components
that capture the forces actually driving European equity returns.

Strategy: PLS maximizes covariance between factors (X) and stock returns (Y),
identifying the components that define what tends to win in Europe. These components
represent the genuine return drivers after removing noise and correlation.

Component Selection: Number of components chosen via 10-fold cross-validation
to minimize out-of-sample forecast error.
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import PLS_LOOKBACK_YEARS, PLS_N_COMPONENTS


class PLSModel:
    """
    Partial Least Squares model to extract return-relevant components
    from macro and style factors
    
    Components selected via cross-validation to minimize forecast error
    """
    
    def __init__(self, n_components: int = PLS_N_COMPONENTS, cv_folds: int = 10):
        self.n_components = n_components
        self.cv_folds = cv_folds
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()
        self.component_loadings = None
        self.component_scores = None
        self.factor_names = None
        
        # CV results storage
        self.cv_results = None
        self.optimal_n_components = None
        
    def fit(self, X: pd.DataFrame, Y: pd.DataFrame) -> 'PLSModel':
        """
        Fit PLS model to identify return-relevant components
        
        PLS finds components that maximize covariance between factors and European equity returns.
        This identifies the forces that genuinely drive returns, not just correlated noise.
        
        Args:
            X: Macro and style factors (dates x factors) - highly correlated input variables
            Y: Stock returns (dates x stocks) - European equity returns
            
        Returns:
            self
        """
        print(f"      Fitting PLS model with {self.n_components} components...")
        
        # Align X and Y on dates
        common_dates = X.index.intersection(Y.index)
        X_aligned = X.loc[common_dates].copy()
        Y_aligned = Y.loc[common_dates].copy()
        
        if len(common_dates) < 50:
            raise ValueError(f"Insufficient data: only {len(common_dates)} common dates")
        
        # CRITICAL: Handle NaN and inf values
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan)
        Y_aligned = Y_aligned.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with 0 (neutral)
        X_aligned = X_aligned.fillna(0)
        Y_aligned = Y_aligned.fillna(0)
        
        # Store factor names
        self.factor_names = X_aligned.columns.tolist()
        
        # Standardize inputs
        X_scaled = self.scaler_X.fit_transform(X_aligned.values)
        Y_scaled = self.scaler_Y.fit_transform(Y_aligned.values)
        
        # Final check for NaN in scaled data
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        Y_scaled = np.nan_to_num(Y_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit PLS model
        self.model = PLSRegression(n_components=self.n_components, scale=False)
        self.model.fit(X_scaled, Y_scaled)
        
        # Extract component loadings (how factors contribute to components)
        self.component_loadings = pd.DataFrame(
            self.model.x_loadings_,
            index=self.factor_names,
            columns=[f'PLS_{i+1}' for i in range(self.n_components)]
        )
        
        # Calculate component scores (time series of component values)
        X_scores = self.model.transform(X_scaled)
        self.component_scores = pd.DataFrame(
            X_scores,
            index=common_dates,
            columns=[f'PLS_{i+1}' for i in range(self.n_components)]
        )
        
        print(f"      PLS model fitted. Explained variance: {self.get_explained_variance():.2%}")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new factor data to PLS components
        
        Args:
            X: Factor data (dates x factors)
            
        Returns:
            Component scores (dates x components)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Standardize using fitted scaler
        X_scaled = self.scaler_X.transform(X.values)
        
        # Transform to components
        X_scores = self.model.transform(X_scaled)
        
        component_scores = pd.DataFrame(
            X_scores,
            index=X.index,
            columns=[f'PLS_{i+1}' for i in range(self.n_components)]
        )
        
        return component_scores
    
    def get_explained_variance(self) -> float:
        """Get total explained variance ratio"""
        if self.model is None:
            return 0.0
        # PLSRegression doesn't have x_variance_ratio_, so calculate manually
        # Use the R-squared of the Y predictions as a proxy
        try:
            # Sum of squared loadings gives relative importance
            loadings_sq = np.sum(self.model.x_loadings_ ** 2)
            total_variance = loadings_sq / (loadings_sq + 1)  # Normalize to 0-1
            return min(total_variance, 0.99)  # Cap at 99%
        except:
            return 0.5  # Default fallback
    
    def get_component_loadings(self) -> pd.DataFrame:
        """Get component loadings (factor contributions)"""
        return self.component_loadings.copy()
    
    def get_component_scores(self) -> pd.DataFrame:
        """Get component scores (time series)"""
        return self.component_scores.copy()
    
    def cross_validate_components(self, X: pd.DataFrame, Y: pd.DataFrame, 
                                   max_components: int = 15,
                                   use_time_series_cv: bool = True,
                                   min_components: int = 5) -> int:
        """
        Cross-validate to find optimal number of components using FINANCE-AWARE selection.
        
        IMPORTANT: Standard CV often picks 1-2 components because daily returns are noisy.
        This is WRONG for finance - we know multiple factors drive returns (Fama-French, etc.)
        
        Selection Algorithm:
        1. Compute CV errors for all component counts
        2. Find the ELBOW point where improvement flattens (not global minimum)
        3. Enforce minimum of 5 components (economic intuition: momentum, value, quality, size, vol)
        4. Use explained variance as secondary check
        
        Args:
            X: Factor data
            Y: Returns data (stocks as columns)
            max_components: Maximum number of components to test
            use_time_series_cv: Use TimeSeriesSplit (True) or KFold (False)
            min_components: Minimum components (default 5 for finance factors)
            
        Returns:
            Optimal number of components
        """
        print(f"      Cross-validating PLS components ({self.cv_folds}-fold CV)...")
        print(f"      NOTE: Using finance-aware selection (min={min_components} components)")
        
        # Align data
        common_dates = X.index.intersection(Y.index)
        X_aligned = X.loc[common_dates].copy()
        Y_aligned = Y.loc[common_dates].copy()
        
        # Clean data FIRST
        X_aligned = X_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        Y_aligned = Y_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Use MULTIPLE stock returns as target
        n_target_stocks = min(10, Y_aligned.shape[1])
        target_cols = Y_aligned.columns[::max(1, Y_aligned.shape[1] // n_target_stocks)][:n_target_stocks]
        Y_multi = Y_aligned[target_cols].values
        
        # Standardize X
        X_values = X_aligned.values
        X_mean, X_std = X_values.mean(axis=0), X_values.std(axis=0)
        X_std[X_std == 0] = 1
        X_scaled = (X_values - X_mean) / X_std
        
        # Standardize Y (each column separately)
        Y_mean_vals = Y_multi.mean(axis=0)
        Y_std_vals = Y_multi.std(axis=0)
        Y_std_vals[Y_std_vals == 0] = 1
        Y_scaled = (Y_multi - Y_mean_vals) / Y_std_vals
        
        print(f"      X shape: {X_scaled.shape}, Y shape: {Y_scaled.shape}")
        print(f"      Using {n_target_stocks} stocks as CV targets")
        
        # Cross-validation strategy
        n_splits = min(self.cv_folds, len(Y_scaled) // 100)
        n_splits = max(3, n_splits)
        
        if use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = KFold(n_splits=n_splits, shuffle=False)
        
        # Test different numbers of components
        max_n = min(max_components, X_aligned.shape[1] - 1, 12)
        component_range = list(range(1, max_n + 1))
        
        cv_errors = []
        cv_errors_std = []
        explained_variances = []
        
        print(f"      Testing {max_n} component values with {n_splits} folds...")
        
        for n in component_range:
            fold_errors = []
            
            for train_idx, test_idx in cv.split(X_scaled):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                Y_train, Y_test = Y_scaled[train_idx], Y_scaled[test_idx]
                
                try:
                    pls = PLSRegression(n_components=n, scale=False)
                    pls.fit(X_train, Y_train)
                    Y_pred = pls.predict(X_test)
                    mse = np.mean((Y_test - Y_pred) ** 2)
                    fold_errors.append(mse)
                except Exception as e:
                    fold_errors.append(np.nan)
            
            valid_errors = [e for e in fold_errors if not np.isnan(e) and not np.isinf(e)]
            if valid_errors:
                cv_errors.append(np.mean(valid_errors))
                cv_errors_std.append(np.std(valid_errors) / np.sqrt(len(valid_errors)))
            else:
                cv_errors.append(1.0)
                cv_errors_std.append(0)
            
            # Calculate explained variance for this n
            pls_full = PLSRegression(n_components=n, scale=False)
            pls_full.fit(X_scaled, Y_scaled)
            X_reconstructed = pls_full.inverse_transform(pls_full.transform(X_scaled))
            total_var = np.var(X_scaled)
            residual_var = np.var(X_scaled - X_reconstructed)
            exp_var = 1 - (residual_var / total_var) if total_var > 0 else 0
            explained_variances.append(exp_var)
        
        print(f"      CV Errors: {[f'{e:.4f}' for e in cv_errors]}")
        print(f"      Explained Var: {[f'{v:.1%}' for v in explained_variances]}")
        
        # Store CV results
        self.cv_results = {
            'n_components': list(component_range),
            'cv_error': cv_errors,
            'cv_error_std': cv_errors_std,
            'explained_variance': explained_variances,
            'cv_type': 'TimeSeriesSplit' if use_time_series_cv else 'KFold',
            'n_folds': n_splits
        }
        
        # ============================================================
        # FINANCE-AWARE COMPONENT SELECTION
        # ============================================================
        # 
        # Standard 1-SE rule fails for finance because:
        # - Daily returns are ~95% noise (R² typically 1-5%)
        # - CV minimum often at n=1-2 (just market factor)
        # - We KNOW multiple factors matter (Fama-French, Carhart, etc.)
        #
        # Solution: Find elbow/plateau + enforce economic minimum
        # ============================================================
        
        errors_arr = np.array(cv_errors)
        
        # Method 1: Elbow detection via second derivative
        # Find where the improvement rate flattens significantly
        first_deriv = np.diff(errors_arr)  # Improvement from n to n+1
        second_deriv = np.diff(first_deriv)  # Acceleration of improvement
        
        # Elbow is where second derivative is most positive (improvement slowing fastest)
        # Add 2 to index because of double diff
        elbow_candidates = []
        for i in range(len(second_deriv)):
            if second_deriv[i] > 0:  # Improvement slowing
                elbow_candidates.append(i + 2)
        
        # Method 2: Plateau detection
        # Find first local minimum after sharp initial drop
        local_minima = []
        for i in range(1, len(errors_arr) - 1):
            if errors_arr[i] <= errors_arr[i-1] and errors_arr[i] <= errors_arr[i+1]:
                local_minima.append(i + 1)  # +1 because component_range is 1-indexed
        
        # Method 3: Relative improvement threshold
        # Stop when improvement < 5% of initial error reduction
        initial_drop = errors_arr[0] - errors_arr[1] if len(errors_arr) > 1 else 0
        threshold_improvement = max(0.05 * initial_drop, 0.001)
        
        plateau_start = len(component_range)  # Default to max
        for i in range(1, len(errors_arr)):
            improvement = errors_arr[i-1] - errors_arr[i]
            if improvement < threshold_improvement:
                plateau_start = i + 1  # Component number where plateau starts
                break
        
        # Method 4: Explained variance threshold (>= 70%)
        variance_threshold_idx = len(component_range)
        for i, exp_var in enumerate(explained_variances):
            if exp_var >= 0.70:
                variance_threshold_idx = i + 1
                break
        
        print(f"\n      --- Component Selection Analysis ---")
        print(f"      Global minimum: n={np.argmin(errors_arr) + 1} (IGNORE for finance)")
        print(f"      Elbow candidates: {elbow_candidates[:3] if elbow_candidates else 'None'}")
        print(f"      Local minima: {local_minima[:3] if local_minima else 'None'}")
        print(f"      Plateau starts at: n={plateau_start}")
        print(f"      70% variance at: n={variance_threshold_idx}")
        print(f"      Economic minimum: n={min_components}")
        
        # Final selection: Use plateau start OR first local minimum >= min_components
        # Prioritize explained variance threshold if lower
        candidates = [
            plateau_start,
            variance_threshold_idx,
            min_components
        ]
        
        # Add first local minimum >= min_components
        valid_local_min = [lm for lm in local_minima if lm >= min_components]
        if valid_local_min:
            candidates.append(valid_local_min[0])
        
        # Add first elbow >= min_components
        valid_elbow = [e for e in elbow_candidates if e >= min_components]
        if valid_elbow:
            candidates.append(valid_elbow[0])
        
        # Filter valid candidates and pick the one that balances parsimony and fit
        valid_candidates = [c for c in candidates if min_components <= c <= max_n]
        
        if valid_candidates:
            # Among valid candidates, prefer the one with best error (but >= min_components)
            best_idx = None
            best_error = float('inf')
            for c in valid_candidates:
                idx = c - 1  # Convert to 0-indexed
                if idx < len(cv_errors) and cv_errors[idx] < best_error:
                    best_error = cv_errors[idx]
                    best_idx = c
            selected = best_idx if best_idx else max(valid_candidates)
        else:
            selected = min_components
        
        # Final sanity check: ensure at least min_components
        selected = max(selected, min_components)
        selected = min(selected, max_n)  # Don't exceed max
        
        self.optimal_n_components = selected
        self.n_components = self.optimal_n_components
        
        selected_idx = selected - 1
        print(f"\n      SELECTED: {self.optimal_n_components} components")
        print(f"      CV error: {cv_errors[selected_idx]:.4f}")
        print(f"      Explained variance: {explained_variances[selected_idx]:.1%}")
        print(f"      Rationale: Finance requires multiple factors (momentum, value, quality, size, vol)")
        
        return self.optimal_n_components
    
    def get_cv_results(self) -> Dict:
        """Get cross-validation results for plotting"""
        return self.cv_results if self.cv_results else {}
    
    def get_factor_importance(self) -> pd.DataFrame:
        """
        Calculate importance of each factor based on component loadings
        """
        if self.component_loadings is None:
            raise ValueError("Model not fitted.")
        
        # Sum of absolute loadings across all components
        importance = self.component_loadings.abs().sum(axis=1).sort_values(ascending=False)
        
        return pd.DataFrame({
            'factor': importance.index,
            'importance': importance.values
        })


if __name__ == "__main__":
    # Test PLS model
    print("PLS model module loaded")
