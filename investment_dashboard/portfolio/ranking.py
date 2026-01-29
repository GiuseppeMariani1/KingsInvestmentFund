"""
Stock Ranking Module

Ranks stocks by model-implied expected return. Winners are stocks with the most stable
and positive exposure to factors that genuinely drive European returns (after noise
and overfitting are removed by regularization).
"""
import pandas as pd
import numpy as np
from typing import Optional


class StockRanker:
    """Ranks stocks by expected return"""
    
    def __init__(self, long_only: bool = True):
        self.long_only = long_only
        
    def rank_stocks(self, expected_returns: pd.Series, 
                   min_return: Optional[float] = None) -> pd.DataFrame:
        """
        Rank stocks by expected return
        
        Args:
            expected_returns: Series with ticker -> expected return
            min_return: Minimum expected return threshold (for long-only)
            
        Returns:
            DataFrame with columns: ticker, expected_return, rank
        """
        # Convert to DataFrame
        rankings = pd.DataFrame({
            'ticker': expected_returns.index,
            'expected_return': expected_returns.values
        })
        
        # Filter negative returns if long-only
        if self.long_only:
            if min_return is None:
                min_return = 0.0
            rankings = rankings[rankings['expected_return'] >= min_return]
        
        # Sort by expected return (descending)
        rankings = rankings.sort_values('expected_return', ascending=False)
        rankings['rank'] = range(1, len(rankings) + 1)
        rankings = rankings.reset_index(drop=True)
        
        return rankings
    
    def get_top_n(self, expected_returns: pd.Series, n: int = 30) -> pd.Series:
        """
        Get top N stocks by expected return
        
        Args:
            expected_returns: Series with expected returns
            n: Number of top stocks to return
            
        Returns:
            Series with top N stocks
        """
        rankings = self.rank_stocks(expected_returns)
        top_n_tickers = rankings.head(n)['ticker'].tolist()
        return expected_returns[top_n_tickers]


if __name__ == "__main__":
    # Test ranking
    print("Stock ranking module loaded")
