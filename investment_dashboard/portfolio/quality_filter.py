"""
PILLAR 1: FUNDAMENTALS
======================
Quality-based stock selection filter.

Fundamentals are SLOW-MOVING ANCHORS, not trading signals.
We use them to define WHAT TO OWN, not WHEN to trade.

This filter is applied BEFORE the regime model runs.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class QualityThresholds:
    """
    Quality thresholds for stock selection.
    
    These are the minimum standards for inclusion in the investable universe.
    Based on academic research and practitioner experience.
    """
    # Profitability - minimum ROE for quality companies
    min_roe: float = 0.05  # 5% minimum ROE
    
    # Balance sheet strength - maximum leverage
    max_debt_equity: float = 3.0  # D/E ratio cap
    
    # Profitability margin - minimum profit margin
    min_profit_margin: float = 0.0  # Allow break-even, exclude loss-makers
    
    # Size filter - minimum for liquidity
    min_market_cap_percentile: float = 0.10  # Bottom 10% excluded
    
    # Enable/disable each filter
    use_roe_filter: bool = True
    use_leverage_filter: bool = True
    use_margin_filter: bool = True
    use_size_filter: bool = True


class QualityFilter:
    """
    Implements Pillar 1: Fundamentals-based stock selection.
    
    The quality filter creates a "quality score" for each stock and
    filters out companies that don't meet minimum standards.
    
    Key principle: Fundamentals define WHAT we want to own.
    They don't tell us WHEN to trade or HOW MUCH.
    """
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()
        self.quality_scores = None
        self.filtered_universe = None
        self.filter_stats = {}
        
    def calculate_quality_scores(self, fundamentals: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate quality scores for each stock.
        
        Args:
            fundamentals: DataFrame with columns: roe, debt_equity, profit_margin, market_cap
                         Index should be ticker symbols
        
        Returns:
            DataFrame with quality scores and component scores
        """
        scores = pd.DataFrame(index=fundamentals.index)
        
        # =====================================================================
        # 1. PROFITABILITY SCORE (0-100)
        # Higher ROE = better quality
        # =====================================================================
        if 'roe' in fundamentals.columns:
            roe = fundamentals['roe'].fillna(0)
            # Cap extreme values
            roe = roe.clip(-0.5, 1.0)
            # Scale to 0-100: ROE of 0% = 0, ROE of 20%+ = 100
            scores['profitability_score'] = (roe.clip(0, 0.20) / 0.20 * 100).round(1)
            scores['roe'] = roe
        else:
            scores['profitability_score'] = 50  # Neutral if missing
            scores['roe'] = np.nan
            
        # =====================================================================
        # 2. BALANCE SHEET SCORE (0-100)
        # Lower debt/equity = better quality
        # =====================================================================
        if 'debt_equity' in fundamentals.columns:
            de = fundamentals['debt_equity'].fillna(1.0)  # Assume moderate if missing
            # Cap extreme values
            de = de.clip(0, 5.0)
            # Invert: Low D/E = high score
            # D/E of 0 = 100, D/E of 2+ = 0
            scores['balance_sheet_score'] = ((2.0 - de.clip(0, 2.0)) / 2.0 * 100).round(1)
            scores['debt_equity'] = de
        else:
            scores['balance_sheet_score'] = 50
            scores['debt_equity'] = np.nan
            
        # =====================================================================
        # 3. MARGIN SCORE (0-100)
        # Higher profit margin = better quality
        # =====================================================================
        if 'profit_margin' in fundamentals.columns:
            pm = fundamentals['profit_margin'].fillna(0)
            # Cap extreme values
            pm = pm.clip(-0.5, 0.5)
            # Scale: margin of 0% = 50, margin of 20%+ = 100, margin of -20% = 0
            scores['margin_score'] = ((pm.clip(-0.20, 0.20) + 0.20) / 0.40 * 100).round(1)
            scores['profit_margin'] = pm
        else:
            scores['margin_score'] = 50
            scores['profit_margin'] = np.nan
            
        # =====================================================================
        # 4. COMPOSITE QUALITY SCORE
        # Weighted average of components
        # =====================================================================
        weights = {
            'profitability_score': 0.40,  # ROE most important
            'balance_sheet_score': 0.35,  # Leverage matters
            'margin_score': 0.25          # Margins supportive
        }
        
        scores['quality_score'] = (
            scores['profitability_score'] * weights['profitability_score'] +
            scores['balance_sheet_score'] * weights['balance_sheet_score'] +
            scores['margin_score'] * weights['margin_score']
        ).round(1)
        
        # Quality tier (for portfolio construction)
        scores['quality_tier'] = pd.cut(
            scores['quality_score'],
            bins=[0, 40, 60, 80, 100],
            labels=['Low', 'Medium', 'High', 'Premium']
        )
        
        self.quality_scores = scores
        return scores
    
    def apply_filters(self, fundamentals: pd.DataFrame) -> Tuple[List[str], Dict]:
        """
        Apply quality filters to create investable universe.
        
        Args:
            fundamentals: DataFrame with fundamental data
            
        Returns:
            Tuple of (list of passing tickers, filter statistics)
        """
        n_total = len(fundamentals)
        passing = pd.Series(True, index=fundamentals.index)
        stats = {'initial': n_total}
        
        # =====================================================================
        # FILTER 1: Minimum ROE
        # =====================================================================
        if self.thresholds.use_roe_filter and 'roe' in fundamentals.columns:
            roe_pass = fundamentals['roe'] >= self.thresholds.min_roe
            # Allow NaN to pass (don't exclude if data missing)
            roe_pass = roe_pass | fundamentals['roe'].isna()
            
            n_fail = (~roe_pass).sum()
            passing = passing & roe_pass
            stats['roe_filter'] = {
                'threshold': f'>= {self.thresholds.min_roe:.1%}',
                'failed': int(n_fail),
                'remaining': int(passing.sum())
            }
            
        # =====================================================================
        # FILTER 2: Maximum Leverage
        # =====================================================================
        if self.thresholds.use_leverage_filter and 'debt_equity' in fundamentals.columns:
            lev_pass = fundamentals['debt_equity'] <= self.thresholds.max_debt_equity
            # Allow NaN to pass
            lev_pass = lev_pass | fundamentals['debt_equity'].isna()
            
            n_fail = (passing & ~lev_pass).sum()
            passing = passing & lev_pass
            stats['leverage_filter'] = {
                'threshold': f'<= {self.thresholds.max_debt_equity:.1f}x',
                'failed': int(n_fail),
                'remaining': int(passing.sum())
            }
            
        # =====================================================================
        # FILTER 3: Minimum Profit Margin
        # =====================================================================
        if self.thresholds.use_margin_filter and 'profit_margin' in fundamentals.columns:
            margin_pass = fundamentals['profit_margin'] >= self.thresholds.min_profit_margin
            # Allow NaN to pass
            margin_pass = margin_pass | fundamentals['profit_margin'].isna()
            
            n_fail = (passing & ~margin_pass).sum()
            passing = passing & margin_pass
            stats['margin_filter'] = {
                'threshold': f'>= {self.thresholds.min_profit_margin:.1%}',
                'failed': int(n_fail),
                'remaining': int(passing.sum())
            }
            
        stats['final'] = int(passing.sum())
        stats['filtered_out'] = n_total - stats['final']
        stats['pass_rate'] = stats['final'] / n_total if n_total > 0 else 0
        
        self.filter_stats = stats
        self.filtered_universe = fundamentals.index[passing].tolist()
        
        return self.filtered_universe, stats
    
    def get_quality_weights(self, tickers: List[str]) -> Dict[str, float]:
        """
        Get quality-based weights for portfolio construction.
        
        Higher quality stocks get higher weight multipliers.
        This is used in Pillar 3 (Risk) to adjust position sizes.
        
        Returns:
            Dict mapping ticker -> quality weight multiplier (0.5 to 1.5)
        """
        if self.quality_scores is None:
            return {t: 1.0 for t in tickers}
            
        weights = {}
        for ticker in tickers:
            if ticker in self.quality_scores.index:
                score = self.quality_scores.loc[ticker, 'quality_score']
                # Map score to weight: 0->0.5, 50->1.0, 100->1.5
                weight = 0.5 + (score / 100)
                weights[ticker] = weight
            else:
                weights[ticker] = 1.0  # Neutral if missing
                
        return weights
    
    def get_summary(self) -> Dict:
        """Get summary of quality filtering."""
        if self.quality_scores is None:
            return {}
            
        return {
            'total_scored': len(self.quality_scores),
            'avg_quality_score': self.quality_scores['quality_score'].mean(),
            'quality_distribution': self.quality_scores['quality_tier'].value_counts().to_dict(),
            'avg_roe': self.quality_scores['roe'].mean(),
            'avg_debt_equity': self.quality_scores['debt_equity'].mean(),
            'avg_profit_margin': self.quality_scores['profit_margin'].mean(),
            'filter_stats': self.filter_stats
        }


def create_quality_filtered_universe(
    all_tickers: List[str],
    fundamentals: pd.DataFrame,
    thresholds: Optional[QualityThresholds] = None
) -> Tuple[List[str], pd.DataFrame, Dict]:
    """
    Convenience function to create a quality-filtered universe.
    
    Args:
        all_tickers: Full list of potential tickers
        fundamentals: DataFrame with fundamental data (index=ticker)
        thresholds: Quality thresholds (optional, uses defaults)
        
    Returns:
        Tuple of:
        - List of tickers passing quality filters
        - DataFrame of quality scores
        - Dict of filter statistics
    """
    qf = QualityFilter(thresholds)
    
    # Calculate scores
    scores = qf.calculate_quality_scores(fundamentals)
    
    # Apply filters
    passing_tickers, stats = qf.apply_filters(fundamentals)
    
    # Only return tickers that were in original list
    final_tickers = [t for t in passing_tickers if t in all_tickers]
    
    return final_tickers, scores, stats


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Example with dummy data
    import numpy as np
    
    tickers = ['AAPL', 'MSFT', 'GOOG', 'META', 'NVDA', 'TSLA', 'BAD1', 'BAD2']
    
    fundamentals = pd.DataFrame({
        'roe': [0.25, 0.30, 0.20, 0.15, 0.35, 0.10, -0.05, 0.02],
        'debt_equity': [0.5, 0.3, 0.2, 0.1, 0.4, 2.5, 4.0, 0.8],
        'profit_margin': [0.25, 0.35, 0.22, 0.28, 0.40, 0.08, -0.10, 0.03],
    }, index=tickers)
    
    # Create filter
    qf = QualityFilter()
    
    # Calculate scores
    scores = qf.calculate_quality_scores(fundamentals)
    print("\nQuality Scores:")
    print(scores)
    
    # Apply filters
    passing, stats = qf.apply_filters(fundamentals)
    print(f"\nPassing tickers: {passing}")
    print(f"\nFilter stats: {stats}")
