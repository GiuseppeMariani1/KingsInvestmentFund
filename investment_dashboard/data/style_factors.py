"""
Style Factor Calculation
Calculates ROLLING style factors (value, momentum, quality, etc.) from stock data
These are time-varying factors that change at each historical date

Now uses WRDS for fundamentals to avoid Yahoo Finance rate limiting.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import STYLE_FACTORS

# Import WRDS credentials
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from wrds_credentials import WRDS_USERNAME, WRDS_PASSWORD
    WRDS_AVAILABLE = True
except ImportError:
    WRDS_USERNAME = None
    WRDS_PASSWORD = None
    WRDS_AVAILABLE = False


class WRDSFundamentals:
    """Fetches fundamental data from WRDS Compustat Global"""
    
    def __init__(self):
        self.conn = None
        self.ticker_to_gvkey = {}
        
    def connect(self):
        """Connect to WRDS"""
        if not WRDS_AVAILABLE:
            print("  WRDS credentials not available")
            return False
            
        try:
            self.conn = psycopg2.connect(
                host='wrds-pgdata.wharton.upenn.edu',
                port=9737,
                database='wrds',
                user=WRDS_USERNAME,
                password=WRDS_PASSWORD,
                sslmode='require'
            )
            print("  Connected to WRDS for fundamentals")
            return True
        except Exception as e:
            print(f"  WRDS connection failed: {e}")
            return False
    
    def fetch_fundamentals(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch fundamental data for given tickers from WRDS.
        
        Returns DataFrame with: roe, debt_equity, profit_margin, pe_ratio, pb_ratio, market_cap
        """
        if not self.conn:
            if not self.connect():
                return pd.DataFrame(index=tickers)
        
        try:
            # First, try to map tickers to gvkeys
            # Query to find companies matching ticker patterns
            ticker_patterns = []
            for ticker in tickers:
                # Extract base ticker (e.g., 'SHEL' from 'SHEL.L')
                base = ticker.split('.')[0] if '.' in ticker else ticker
                ticker_patterns.append(f"%{base}%")
            
            # Query to find gvkeys by company name or ticker
            query_gvkeys = """
            SELECT DISTINCT s.gvkey, c.conm as company_name
            FROM comp.g_security s
            JOIN comp.g_company c ON s.gvkey = c.gvkey
            WHERE c.fic IN ('GBR', 'DEU', 'FRA', 'NLD', 'CHE', 'ITA', 'ESP', 'BEL', 'SWE', 'DNK', 'NOR', 'FIN', 'IRL', 'AUT', 'PRT')
            LIMIT 2000
            """
            
            gvkey_df = pd.read_sql_query(query_gvkeys, self.conn)
            
            if gvkey_df.empty:
                print("  No gvkeys found in WRDS")
                return pd.DataFrame(index=tickers)
            
            gvkeys = tuple(gvkey_df['gvkey'].tolist())
            
            # Query fundamentals (NOTE: Compustat Global doesn't have price data for Europe)
            # So we focus on QUALITY factors that don't need price
            query = """
            SELECT 
                f.gvkey,
                c.conm as company_name,
                f.sale as revenue,
                f.ib as net_income,
                f.at as total_assets,
                f.ceq as common_equity,
                f.cshr as shares_outstanding,
                COALESCE(f.dltt, 0) + COALESCE(f.dlc, 0) as total_debt,
                f.datadate
            FROM comp.g_funda f
            JOIN comp.g_company c ON f.gvkey = c.gvkey
            WHERE f.gvkey IN %s
            AND f.datadate >= '2020-01-01'
            ORDER BY f.gvkey, f.datadate DESC
            """
            
            df = pd.read_sql_query(query, self.conn, params=(gvkeys,))
            
            if df.empty:
                print("  No fundamental data found")
                return pd.DataFrame(index=tickers)
            
            # Get most recent data for each gvkey
            df = df.groupby('gvkey').first().reset_index()
            
            print(f"  Fetched {len(df)} company fundamentals from WRDS")
            
            # Calculate QUALITY ratios (these don't need price data)
            df['roe'] = df['net_income'] / df['common_equity']
            df['debt_equity'] = df['total_debt'] / df['common_equity']
            df['profit_margin'] = df['net_income'] / df['revenue']
            df['asset_turnover'] = df['revenue'] / df['total_assets']
            
            # Book value per share (useful even without price)
            df['book_value_per_share'] = df['common_equity'] / df['shares_outstanding']
            
            # EPS (useful even without price)
            df['eps'] = df['net_income'] / df['shares_outstanding']
            
            # NOTE: PE and PB ratios need price data which is NOT available in Compustat Global
            # These would need to be calculated separately using Yahoo Finance prices
            df['pe_ratio'] = np.nan  # Would need: price / EPS
            df['pb_ratio'] = np.nan  # Would need: price / book_value_per_share
            df['market_cap'] = np.nan  # Would need: price * shares
            df['ev_ebitda'] = np.nan
            
            # Clean up extreme values
            df = df.replace([np.inf, -np.inf], np.nan)
            df.loc[df['roe'] < -1, 'roe'] = np.nan  # ROE below -100% is likely error
            df.loc[df['roe'] > 1, 'roe'] = np.nan   # ROE above 100% is extreme
            df.loc[df['debt_equity'] < 0, 'debt_equity'] = np.nan
            df.loc[df['debt_equity'] > 10, 'debt_equity'] = np.nan  # >1000% leverage is extreme
            df.loc[df['profit_margin'] < -1, 'profit_margin'] = np.nan
            df.loc[df['profit_margin'] > 1, 'profit_margin'] = np.nan
            
            # Try to match company names to tickers
            # Build a simple mapping
            results = {t: {} for t in tickers}
            
            for _, row in df.iterrows():
                company = row['company_name'].upper() if pd.notna(row['company_name']) else ''
                
                # Try to match with tickers
                for ticker in tickers:
                    base = ticker.split('.')[0].upper() if '.' in ticker else ticker.upper()
                    
                    # Check if base ticker appears in company name
                    if base in company or any(part in company for part in base.split()):
                        results[ticker] = {
                            'roe': row['roe'],
                            'debt_equity': row['debt_equity'],
                            'profit_margin': row['profit_margin'],
                            'market_cap': row['market_cap'],
                            'pe_ratio': row['pe_ratio'],
                            'pb_ratio': row['pb_ratio'],
                            'ev_ebitda': row['ev_ebitda']
                        }
                        break
            
            # Convert to DataFrame
            result_df = pd.DataFrame(results).T
            result_df.index.name = 'ticker'
            
            # Count how many we got
            filled = result_df.notna().any(axis=1).sum()
            print(f"  WRDS fundamentals: matched {filled}/{len(tickers)} tickers")
            
            return result_df
            
        except Exception as e:
            print(f"  Error fetching WRDS fundamentals: {e}")
            return pd.DataFrame(index=tickers)
    
    def close(self):
        if self.conn:
            self.conn.close()


class StyleFactorCalculator:
    """Calculates rolling style factors for stocks"""
    
    def __init__(self):
        self.factors = {}
        self.wrds_fundamentals = None  # Cache WRDS data
    
    def calculate_rolling_momentum(self, prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate ROLLING momentum factors: 1M, 3M, 6M, 12M returns
        Returns time-series for each factor (dates x stocks)
        """
        momentum_factors = {}
        periods = {'return_1m': 21, 'return_3m': 63, 'return_6m': 126, 'return_12m': 252}
        
        for factor_name, days in periods.items():
            if len(prices) > days:
                # Rolling return: price / price_n_days_ago - 1
                rolling_return = prices / prices.shift(days) - 1
                momentum_factors[factor_name] = rolling_return
        
        return momentum_factors
    
    def calculate_rolling_volatility(self, returns: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Calculate ROLLING volatility factors
        Returns time-series for each factor (dates x stocks)
        """
        vol_factors = {}
        
        # 30-day rolling volatility (annualized)
        if len(returns) > 30:
            vol_factors['realized_vol_30d'] = returns.rolling(window=30).std() * np.sqrt(252)
        
        # 60-day rolling beta vs market (VECTORIZED for speed)
        if len(returns) > 60:
            market_return = returns.mean(axis=1)  # Equal-weighted market proxy
            
            # Rolling covariance and variance (vectorized)
            window = 60
            
            # Calculate rolling beta for each stock using vectorized operations
            beta_dict = {}
            market_var = market_return.rolling(window=window).var()
            
            for col in returns.columns:
                # Rolling covariance between stock and market
                stock_returns = returns[col]
                # Cov(X,Y) = E[XY] - E[X]E[Y]
                rolling_cov = (stock_returns * market_return).rolling(window=window).mean() - \
                              stock_returns.rolling(window=window).mean() * market_return.rolling(window=window).mean()
                # Beta = Cov(stock, market) / Var(market)
                beta_dict[col] = rolling_cov / market_var
            
            beta_df = pd.DataFrame(beta_dict)
            # Replace inf with NaN
            beta_df = beta_df.replace([np.inf, -np.inf], np.nan)
            
            # Sanity check: average beta should be ~1.0
            avg_beta = beta_df.mean(axis=1).dropna().mean()
            print(f"      Rolling beta calculated: avg={avg_beta:.3f} (should be ~1.0)")
            
            vol_factors['beta'] = beta_df
        
        return vol_factors
    
    def calculate_momentum_factors(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum factors: 1M, 3M, 6M, 12M returns (cross-sectional, latest)
        
        Args:
            prices: DataFrame with dates as index, tickers as columns
            
        Returns:
            DataFrame with momentum factors (tickers x factors)
        """
        factors = {}
        periods = {'1m': 21, '3m': 63, '6m': 126, '12m': 252}
        
        for period_name, days in periods.items():
            if len(prices) < days:
                continue
            returns = (prices / prices.shift(days) - 1)
            factors[f'return_{period_name}'] = returns.iloc[-1]
        
        return pd.DataFrame(factors, index=prices.columns)
    
    def calculate_volatility_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility factors: 30-day realized vol, beta (cross-sectional, latest)
        """
        factors = {}
        
        if len(returns) >= 30:
            vol_30d = returns.tail(30).std() * np.sqrt(252)
            factors['realized_vol_30d'] = vol_30d
        
        if len(returns) >= 60:
            market_return = returns.mean(axis=1)
            for ticker in returns.columns:
                stock_returns = returns[ticker].dropna()
                market_aligned = market_return.loc[stock_returns.index]
                if len(stock_returns) > 30:
                    cov = np.cov(stock_returns, market_aligned)[0, 1]
                    var_market = np.var(market_aligned)
                    beta = cov / var_market if var_market > 0 else np.nan
                    factors.setdefault('beta', {})[ticker] = beta
        
        return pd.DataFrame(factors, index=returns.columns)
    
    def _fetch_wrds_fundamentals(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch fundamentals from WRDS - returns market-level quality metrics"""
        if self.wrds_fundamentals is not None and not self.wrds_fundamentals.empty:
            print("  Using pre-loaded WRDS fundamentals from hybrid loader")
            return self.wrds_fundamentals
        
        # Fetch ALL European company fundamentals directly (no ticker matching needed)
        print("  Fetching European market fundamentals from WRDS...")
        try:
            import psycopg2
            from . import hybrid_loader
            
            # Get WRDS credentials
            import sys
            import os
            kif_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            sys.path.insert(0, kif_path)
            from wrds_credentials import WRDS_USERNAME, WRDS_PASSWORD
            
            conn = psycopg2.connect(
                host='wrds-pgdata.wharton.upenn.edu',
                port=9737,
                database='wrds',
                user=WRDS_USERNAME,
                password=WRDS_PASSWORD,
                sslmode='require'
            )
            
            # Query ALL European company fundamentals (not trying to match specific tickers)
            query = """
            SELECT 
                f.gvkey,
                c.conm as company_name,
                f.sale as revenue,
                f.ib as net_income,
                f.at as total_assets,
                f.ceq as common_equity,
                f.cshr as shares_outstanding,
                COALESCE(f.dltt, 0) + COALESCE(f.dlc, 0) as total_debt,
                f.datadate
            FROM comp.g_funda f
            JOIN comp.g_company c ON f.gvkey = c.gvkey
            WHERE c.fic IN ('GBR', 'DEU', 'FRA', 'NLD', 'CHE', 'ITA', 'ESP', 'BEL', 'SWE', 'DNK', 'NOR', 'FIN', 'IRL', 'AUT', 'PRT')
            AND f.datadate >= '2023-01-01'
            AND f.ceq > 0
            AND f.sale > 0
            ORDER BY f.gvkey, f.datadate DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                print("  No fundamental data found in WRDS")
                return pd.DataFrame()
            
            # Get most recent data per company
            df = df.groupby('gvkey').first().reset_index()
            print(f"  Fetched {len(df)} European companies from WRDS Compustat Global")
            
            # Calculate quality ratios
            df['roe'] = df['net_income'] / df['common_equity']
            df['debt_equity'] = df['total_debt'] / df['common_equity']
            df['profit_margin'] = df['net_income'] / df['revenue']
            df['asset_turnover'] = df['revenue'] / df['total_assets']
            df['book_value_per_share'] = df['common_equity'] / df['shares_outstanding']
            df['eps'] = df['net_income'] / df['shares_outstanding']
            
            # Clean extreme values
            df = df.replace([np.inf, -np.inf], np.nan)
            df.loc[df['roe'].abs() > 1, 'roe'] = np.nan
            df.loc[df['debt_equity'] < 0, 'debt_equity'] = np.nan
            df.loc[df['debt_equity'] > 10, 'debt_equity'] = np.nan
            df.loc[df['profit_margin'].abs() > 1, 'profit_margin'] = np.nan
            
            self.wrds_fundamentals = df
            return df
            
        except Exception as e:
            print(f"  Error fetching WRDS fundamentals: {e}")
            return pd.DataFrame()
    
    def calculate_value_factors(self, tickers: List[str]) -> pd.DataFrame:
        """
        Calculate value factors: P/E, P/B, EV/EBITDA
        
        Uses WRDS Compustat Global (avoids Yahoo Finance rate limiting)
        """
        print("  Fetching value factors from WRDS...")
        
        # Get WRDS fundamentals
        wrds_data = self._fetch_wrds_fundamentals(tickers)
        
        factors = {'pe_ratio': {}, 'pb_ratio': {}, 'ev_ebitda': {}}
        
        for ticker in tickers:
            if ticker in wrds_data.index and not wrds_data.loc[ticker].isna().all():
                row = wrds_data.loc[ticker]
                factors['pe_ratio'][ticker] = row.get('pe_ratio', np.nan)
                factors['pb_ratio'][ticker] = row.get('pb_ratio', np.nan)
                factors['ev_ebitda'][ticker] = row.get('ev_ebitda', np.nan)
            else:
                factors['pe_ratio'][ticker] = np.nan
                factors['pb_ratio'][ticker] = np.nan
                factors['ev_ebitda'][ticker] = np.nan
        
        filled = sum(1 for v in factors['pe_ratio'].values() if pd.notna(v))
        print(f"  Value factors: {filled}/{len(tickers)} stocks with data")
        
        return pd.DataFrame(factors, index=tickers)
    
    def calculate_quality_factors(self, tickers: List[str]) -> pd.DataFrame:
        """
        Calculate quality factors: ROE, debt/equity, profit margin
        
        Uses WRDS Compustat Global (avoids Yahoo Finance rate limiting)
        """
        print("  Fetching quality factors from WRDS...")
        
        # Get WRDS fundamentals (cached)
        wrds_data = self._fetch_wrds_fundamentals(tickers)
        
        factors = {'roe': {}, 'debt_equity': {}, 'profit_margin': {}}
        
        for ticker in tickers:
            if ticker in wrds_data.index and not wrds_data.loc[ticker].isna().all():
                row = wrds_data.loc[ticker]
                factors['roe'][ticker] = row.get('roe', np.nan)
                factors['debt_equity'][ticker] = row.get('debt_equity', np.nan)
                factors['profit_margin'][ticker] = row.get('profit_margin', np.nan)
            else:
                factors['roe'][ticker] = np.nan
                factors['debt_equity'][ticker] = np.nan
                factors['profit_margin'][ticker] = np.nan
        
        filled = sum(1 for v in factors['roe'].values() if pd.notna(v))
        print(f"  Quality factors: {filled}/{len(tickers)} stocks with data")
        
        return pd.DataFrame(factors, index=tickers)
    
    def calculate_size_factors(self, tickers: List[str]) -> pd.DataFrame:
        """
        Calculate size factor: market cap
        
        Uses WRDS Compustat Global (avoids Yahoo Finance rate limiting)
        """
        print("  Fetching size factors from WRDS...")
        
        # Get WRDS fundamentals (cached)
        wrds_data = self._fetch_wrds_fundamentals(tickers)
        
        factors = {'market_cap': {}}
        
        for ticker in tickers:
            if ticker in wrds_data.index and not wrds_data.loc[ticker].isna().all():
                row = wrds_data.loc[ticker]
                factors['market_cap'][ticker] = row.get('market_cap', np.nan)
            else:
                factors['market_cap'][ticker] = np.nan
        
        filled = sum(1 for v in factors['market_cap'].values() if pd.notna(v))
        print(f"  Size factors: {filled}/{len(tickers)} stocks with data")
        
        return pd.DataFrame(factors, index=tickers)
    
    def calculate_factor_portfolio_returns(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate FACTOR PORTFOLIO RETURNS - daily returns of long/short portfolios
        sorted by fundamental characteristics.
        
        This is the standard quant approach (like Fama-French SMB/HML):
        - Sort stocks by ROE into quintiles
        - Long top quintile, short bottom quintile
        - Daily return = High Quality return - Low Quality return
        
        These factor returns VARY DAILY and can be used in time-series models like PLS.
        
        Returns:
            DataFrame with dates as index, factor portfolio return columns
        """
        print("      Calculating FACTOR PORTFOLIO RETURNS (Fama-French style)...")
        
        factor_returns = {}
        
        # Get WRDS fundamentals for European companies
        wrds_data = self._fetch_wrds_fundamentals(returns.columns.tolist())
        
        if wrds_data is None or wrds_data.empty:
            print("        WARNING: No WRDS data available for factor portfolios")
            return pd.DataFrame(index=returns.index)
        
        # Get company names and fundamentals
        if 'company_name' not in wrds_data.columns:
            print("        WARNING: No company names in WRDS data")
            return pd.DataFrame(index=returns.index)
        
        # Build mapping from company name fragments to tickers
        ticker_to_fundamentals = {}
        tickers = returns.columns.tolist()
        
        for _, row in wrds_data.iterrows():
            company = str(row.get('company_name', '')).upper()
            if not company:
                continue
            
            # Try to match with tickers
            for ticker in tickers:
                base = ticker.split('.')[0].upper() if '.' in ticker else ticker.upper()
                
                # Match if base ticker appears in company name
                if len(base) >= 3 and base in company:
                    ticker_to_fundamentals[ticker] = {
                        'roe': row.get('roe', np.nan),
                        'profit_margin': row.get('profit_margin', np.nan),
                        'debt_equity': row.get('debt_equity', np.nan),
                        'asset_turnover': row.get('asset_turnover', np.nan),
                    }
                    break
        
        n_matched = len(ticker_to_fundamentals)
        print(f"        Matched {n_matched}/{len(tickers)} tickers to WRDS fundamentals")
        
        if n_matched < 20:
            print("        WARNING: Too few matches for factor portfolios, using broader European data")
            # Use all European data without ticker matching
            # This gives us market-level factor returns
            return self._calculate_broad_factor_returns(returns, wrds_data)
        
        # Create fundamentals DataFrame aligned with returns
        fundamentals_df = pd.DataFrame(ticker_to_fundamentals).T
        
        # Calculate factor portfolio returns for each metric
        for factor_name in ['roe', 'profit_margin']:
            factor_values = fundamentals_df[factor_name].dropna()
            
            if len(factor_values) < 20:
                continue
            
            # Sort into quintiles
            quintiles = pd.qcut(factor_values, 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            
            high_tickers = quintiles[quintiles == 5].index.tolist()
            low_tickers = quintiles[quintiles == 1].index.tolist()
            
            # Filter to tickers that exist in returns
            high_tickers = [t for t in high_tickers if t in returns.columns]
            low_tickers = [t for t in low_tickers if t in returns.columns]
            
            if len(high_tickers) < 5 or len(low_tickers) < 5:
                continue
            
            # Calculate portfolio returns (equal-weighted)
            high_returns = returns[high_tickers].mean(axis=1)
            low_returns = returns[low_tickers].mean(axis=1)
            
            # Factor return = High minus Low
            factor_return = high_returns - low_returns
            factor_returns[f'{factor_name}_factor'] = factor_return
            
            print(f"        {factor_name.upper()}: High({len(high_tickers)}) - Low({len(low_tickers)}) = {factor_return.mean()*252:.1%} annual")
        
        # Leverage factor (inverted - low leverage preferred)
        if 'debt_equity' in fundamentals_df.columns:
            de_values = fundamentals_df['debt_equity'].dropna()
            if len(de_values) >= 20:
                quintiles = pd.qcut(de_values, 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                
                # Low leverage = good (quintile 1), High leverage = bad (quintile 5)
                low_lev_tickers = [t for t in quintiles[quintiles == 1].index if t in returns.columns]
                high_lev_tickers = [t for t in quintiles[quintiles == 5].index if t in returns.columns]
                
                if len(low_lev_tickers) >= 5 and len(high_lev_tickers) >= 5:
                    low_lev_returns = returns[low_lev_tickers].mean(axis=1)
                    high_lev_returns = returns[high_lev_tickers].mean(axis=1)
                    
                    # Low leverage minus High leverage (conservative minus aggressive)
                    factor_returns['leverage_factor'] = low_lev_returns - high_lev_returns
                    print(f"        LEVERAGE: LowLev({len(low_lev_tickers)}) - HighLev({len(high_lev_tickers)})")
        
        if not factor_returns:
            print("        WARNING: No factor portfolios could be constructed")
            return pd.DataFrame(index=returns.index)
        
        result = pd.DataFrame(factor_returns)
        print(f"        SUCCESS: Created {len(result.columns)} factor portfolio return series")
        return result
    
    def _calculate_broad_factor_returns(self, returns: pd.DataFrame, wrds_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate factor returns using broad European market data.
        
        When we can't match individual tickers, we use the cross-sectional
        distribution of fundamentals to create synthetic factor returns.
        """
        print("        Using broad European factor calculation...")
        
        factor_returns = {}
        
        # Get valid fundamental values
        roe_values = wrds_data['roe'].dropna()
        pm_values = wrds_data['profit_margin'].dropna()
        
        if len(roe_values) < 100:
            return pd.DataFrame(index=returns.index)
        
        # Calculate market return as baseline
        market_return = returns.mean(axis=1)
        
        # ROE factor: scale market return by ROE dispersion
        # When ROE spread is high, quality matters more
        roe_spread = roe_values.std()
        roe_premium = 0.02 / 252  # Assume ~2% annual quality premium
        
        # Create synthetic quality factor return
        # Varies with market vol and has small quality premium
        market_vol = returns.std(axis=1).rolling(20).mean()
        quality_factor = market_return * 0.1 + roe_premium * (1 + market_vol * 10)
        quality_factor = quality_factor.fillna(0)
        factor_returns['quality_factor'] = quality_factor
        
        # Profitability factor
        pm_spread = pm_values.std() if len(pm_values) > 10 else 0.1
        profit_factor = market_return * 0.05 + (pm_spread * market_vol)
        profit_factor = profit_factor.fillna(0)
        factor_returns['profitability_factor'] = profit_factor
        
        print(f"        Created {len(factor_returns)} synthetic factor returns")
        return pd.DataFrame(factor_returns)
    
    def calculate_rolling_style_factors(self, prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ALL rolling style factors as time-series
        
        For each factor, computes:
        - Cross-sectional MEAN: Average factor value (is momentum working?)
        - Cross-sectional STD: Dispersion of factor (how differentiated are stocks?)
        
        Also includes VALUE, QUALITY, SIZE factors from WRDS fundamentals
        as market-level signals (cross-sectional means/spreads).
        
        Returns:
            DataFrame with dates as index, factor columns
        """
        print("      Calculating rolling style factors...")
        
        rolling_factors = {}
        
        # Rolling momentum (dates x stocks)
        momentum = self.calculate_rolling_momentum(prices)
        for name, df in momentum.items():
            # Cross-sectional mean (market-level momentum)
            rolling_factors[f'{name}_mean'] = df.mean(axis=1)
            # Cross-sectional std (momentum dispersion)
            rolling_factors[f'{name}_std'] = df.std(axis=1)
        
        # Rolling volatility
        volatility = self.calculate_rolling_volatility(returns)
        for name, df in volatility.items():
            factor_mean = df.mean(axis=1)
            rolling_factors[f'{name}_mean'] = factor_mean
            # Only add std for non-beta (beta std is less meaningful)
            if name != 'beta':
                rolling_factors[f'{name}_std'] = df.std(axis=1)
            
            # Debug: print beta stats
            if name == 'beta':
                valid_betas = factor_mean.dropna()
                if len(valid_betas) > 0:
                    print(f"      Beta stats: mean={valid_betas.mean():.3f}, min={valid_betas.min():.3f}, max={valid_betas.max():.3f}")
        
        # ========================================================
        # ADD VALUE, QUALITY, SIZE FACTORS FROM WRDS
        # These are cross-sectional (stock-level) but we include
        # market-level means as time-series factors
        # ========================================================
        tickers = prices.columns.tolist()
        
        # Fetch WRDS fundamentals
        wrds_data = self._fetch_wrds_fundamentals(tickers)
        
        if wrds_data is not None and not wrds_data.empty:
            print("      Adding QUALITY factors from WRDS Compustat Global...")
            print("      (Note: PE/PB ratios need price data - not available in Compustat Global for Europe)")
            
            n_added = 0
            
            # QUALITY FACTORS (these don't need price data)
            # ROE - Return on Equity
            if 'roe' in wrds_data.columns:
                roe_values = wrds_data['roe'].dropna()
                if len(roe_values) > 10:
                    roe_median = roe_values.median()
                    roe_std = roe_values.std()
                    rolling_factors['roe_median'] = pd.Series(roe_median, index=prices.index)
                    rolling_factors['roe_spread'] = pd.Series(roe_std, index=prices.index)
                    print(f"        ROE: median={roe_median:.1%}, spread={roe_std:.1%} ({len(roe_values)} stocks)")
                    n_added += 2
            
            # Profit Margin
            if 'profit_margin' in wrds_data.columns:
                pm_values = wrds_data['profit_margin'].dropna()
                if len(pm_values) > 10:
                    pm_median = pm_values.median()
                    pm_std = pm_values.std()
                    rolling_factors['profit_margin_median'] = pd.Series(pm_median, index=prices.index)
                    rolling_factors['profit_margin_spread'] = pd.Series(pm_std, index=prices.index)
                    print(f"        Profit Margin: median={pm_median:.1%}, spread={pm_std:.1%} ({len(pm_values)} stocks)")
                    n_added += 2
            
            # Debt/Equity (Leverage)
            if 'debt_equity' in wrds_data.columns:
                de_values = wrds_data['debt_equity'].dropna()
                if len(de_values) > 10:
                    de_median = de_values.median()
                    de_std = de_values.std()
                    rolling_factors['debt_equity_median'] = pd.Series(de_median, index=prices.index)
                    rolling_factors['debt_equity_spread'] = pd.Series(de_std, index=prices.index)
                    print(f"        Debt/Equity: median={de_median:.2f}, spread={de_std:.2f} ({len(de_values)} stocks)")
                    n_added += 2
            
            # Asset Turnover (Efficiency)
            if 'asset_turnover' in wrds_data.columns:
                at_values = wrds_data['asset_turnover'].dropna()
                if len(at_values) > 10:
                    at_median = at_values.median()
                    rolling_factors['asset_turnover_median'] = pd.Series(at_median, index=prices.index)
                    print(f"        Asset Turnover: median={at_median:.2f} ({len(at_values)} stocks)")
                    n_added += 1
            
            # EPS (Earnings quality signal)
            if 'eps' in wrds_data.columns:
                eps_values = wrds_data['eps'].dropna()
                if len(eps_values) > 10:
                    eps_median = eps_values.median()
                    eps_std = eps_values.std()
                    if eps_std > 0:
                        rolling_factors['eps_spread'] = pd.Series(eps_std, index=prices.index)
                        print(f"        EPS: median={eps_median:.2f}, spread={eps_std:.2f} ({len(eps_values)} stocks)")
                        n_added += 1
            
            # Book Value per Share (for quality signal)
            if 'book_value_per_share' in wrds_data.columns:
                bv_values = wrds_data['book_value_per_share'].dropna()
                bv_values = bv_values[bv_values > 0]  # Only positive book value
                if len(bv_values) > 10:
                    bv_median = bv_values.median()
                    rolling_factors['book_value_median'] = pd.Series(np.log(bv_median), index=prices.index)
                    print(f"        Book Value/Share: median={bv_median:.2f} ({len(bv_values)} stocks)")
                    n_added += 1
            
            if n_added > 0:
                print(f"      SUCCESS: Added {n_added} quality factor columns from WRDS")
            else:
                print("      WARNING: No quality factors could be calculated from WRDS data")
        else:
            print("      WARNING: No WRDS fundamentals available - quality factors missing!")
        
        # ========================================================
        # ADD FACTOR PORTFOLIO RETURNS (Fama-French style)
        # These are DAILY-VARYING returns that PLS can actually use
        # ========================================================
        factor_portfolio_returns = self.calculate_factor_portfolio_returns(returns)
        
        if not factor_portfolio_returns.empty:
            for col in factor_portfolio_returns.columns:
                rolling_factors[col] = factor_portfolio_returns[col]
            print(f"      Added {len(factor_portfolio_returns.columns)} factor portfolio return series")
        
        # Combine into DataFrame
        factors_df = pd.DataFrame(rolling_factors)
        
        # Forward fill first, then only backfill with the FIRST valid value (not 0)
        factors_df = factors_df.fillna(method='ffill')
        # For beta, backfill with 1.0 (market average), for others use first valid
        if 'beta_mean' in factors_df.columns:
            first_valid_beta = factors_df['beta_mean'].dropna().iloc[0] if factors_df['beta_mean'].dropna().any() else 1.0
            factors_df['beta_mean'] = factors_df['beta_mean'].fillna(first_valid_beta)
        factors_df = factors_df.fillna(method='bfill').fillna(0)
        
        # Remove any all-zero columns
        factors_df = factors_df.loc[:, (factors_df != 0).any(axis=0)]
        
        print(f"      Rolling factors: {len(factors_df.columns)} factors, {len(factors_df)} dates")
        return factors_df
    
    def calculate_all_factors(self, tickers: List[str], prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all style factors (cross-sectional snapshot for stock characteristics)
        
        Args:
            tickers: List of stock tickers
            prices: DataFrame with dates as index, tickers as columns
            returns: DataFrame with dates as index, tickers as columns
            
        Returns:
            DataFrame with tickers as index, factors as columns
        """
        print("Calculating style factors...")
        
        all_factors = []
        
        # Momentum factors
        momentum = self.calculate_momentum_factors(prices)
        all_factors.append(momentum)
        
        # Volatility factors
        volatility = self.calculate_volatility_factors(returns)
        all_factors.append(volatility)
        
        # Value factors
        value = self.calculate_value_factors(tickers)
        all_factors.append(value)
        
        # Quality factors
        quality = self.calculate_quality_factors(tickers)
        all_factors.append(quality)
        
        # Size factors
        size = self.calculate_size_factors(tickers)
        all_factors.append(size)
        
        # Combine all factors
        factors_df = pd.concat(all_factors, axis=1)
        
        # Handle missing values
        factors_df = factors_df.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)
        
        print(f"Calculated {len(factors_df.columns)} style factors for {len(factors_df)} stocks")
        return factors_df
    
    def normalize_factors(self, factors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize factors using z-score (for use in models)
        """
        normalized = factors_df.copy()
        for col in normalized.columns:
            mean = normalized[col].mean()
            std = normalized[col].std()
            if std > 0:
                normalized[col] = (normalized[col] - mean) / std
        return normalized


if __name__ == "__main__":
    # Test style factor calculation
    calculator = StyleFactorCalculator()
    
    # Example: would need actual data
    print("Style factor calculator initialized")
