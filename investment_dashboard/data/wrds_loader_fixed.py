"""
WRDS Data Loader - Fixed Version
Uses direct PostgreSQL connection to avoid WRDS library issues
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Import credentials
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from wrds_credentials import WRDS_USERNAME, WRDS_PASSWORD
except ImportError:
    WRDS_USERNAME = None
    WRDS_PASSWORD = None


class WRDSEquityUniverse:
    """Manages European equity universe using direct WRDS PostgreSQL connection"""
    
    def __init__(self, top_n: int = 500):
        self.top_n = top_n
        self.stocks = []
        self.stock_data = {}
        self.conn = None
        
    def connect(self):
        """Connect to WRDS PostgreSQL database directly"""
        print("Connecting to WRDS (direct PostgreSQL connection)...")
        try:
            self.conn = psycopg2.connect(
                host='wrds-pgdata.wharton.upenn.edu',
                port=9737,
                database='wrds',
                user=WRDS_USERNAME if WRDS_USERNAME else 'mattyyychan',
                password=WRDS_PASSWORD if WRDS_PASSWORD else 'Mc2003uk!!!!'
            )
            print("Connected to WRDS successfully!")
        except Exception as e:
            print(f"Error connecting to WRDS: {e}")
            raise
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        if self.conn is None:
            self.connect()
        return pd.read_sql_query(sql, self.conn)
    
    def build_universe(self) -> pd.DataFrame:
        """
        Build European equity universe from Thomson Reuters Worldscope
        """
        if self.conn is None:
            self.connect()
        
        print(f"Building universe of top {self.top_n} European stocks from WRDS Compustat Global...")
        
        # Query Compustat Global for European stocks
        # For now, get companies first, then we'll calculate market cap from daily prices
        query = f"""
        SELECT DISTINCT
            c.gvkey,
            c.conm as company_name,
            c.loc as country,
            c.gsector,
            c.gind
        FROM comp.g_company c
        WHERE c.loc IN ('AUT', 'BEL', 'CHE', 'DEU', 'DNK', 'ESP', 
                        'FIN', 'FRA', 'GBR', 'IRL', 'ITA', 'NLD', 
                        'NOR', 'SWE')
        AND c.costat = 'A'  -- Active companies only
        LIMIT {self.top_n * 3}  -- Get more, will filter by availability later
        """
        
        try:
            df = self.query(query)
            
            if df.empty:
                raise ValueError("No European stocks found in WRDS. Check database access.")
            
            # Use gvkey as stock identifier
            self.stocks = df['gvkey'].tolist()
            self.stock_data = df.set_index('gvkey').to_dict('index')
            
            print(f"Universe built: {len(self.stocks)} European stocks")
            print(f"\nSample companies:")
            for i, row in df.head(5).iterrows():
                print(f"  {i+1}. {row['company_name']} ({row['country']}) - {row['gvkey']}")
            
            return df
            
        except Exception as e:
            print(f"Error querying WRDS: {e}")
            print("\nIf you see 'relation does not exist', you may not have access to Compustat Global.")
            print("Contact King's College library to verify WRDS database subscriptions.")
            raise
    
    def fetch_returns(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch stock returns from Compustat Global Daily (g_company_daily)
        Uses gvkey directly - no ticker mapping needed
        """
        if not self.stocks:
            raise ValueError("Universe not built. Call build_universe() first.")
        
        if self.conn is None:
            self.connect()
        
        print(f"Fetching returns from WRDS Compustat Global ({start_date.date()} to {end_date.date()})...")
        print(f"  Querying data for {len(self.stocks)} stocks...")
        
        # Convert gvkeys to tuple
        gvkeys_tuple = tuple(self.stocks)
        
        # Query from comp.g_company_daily - has prices by gvkey
        query = """
        SELECT 
            datadate as date_,
            gvkey,
            prccd as price_close
        FROM comp.g_company_daily
        WHERE gvkey IN %s
        AND datadate >= %s
        AND datadate <= %s
        AND prccd IS NOT NULL
        AND prccd > 0
        ORDER BY gvkey, datadate
        """
        
        try:
            df = pd.read_sql_query(
                query, 
                self.conn,
                params=(gvkeys_tuple, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            )
            
            if df.empty:
                raise ValueError("No return data fetched from WRDS Compustat Global.")
            
            # Calculate returns
            df['date'] = pd.to_datetime(df['date_'])
            df = df.sort_values(['gvkey', 'date'])
            
            # Calculate daily returns from prices
            df['return'] = df.groupby('gvkey')['price_close'].pct_change()
            
            # Pivot to wide format (gvkey as column names)
            returns_wide = df.pivot(index='date', columns='gvkey', values='return')
            returns_wide = returns_wide.dropna(how='all', axis=1)
            
            print(f"Fetched returns for {len(returns_wide.columns)} stocks over {len(returns_wide)} days")
            
            return returns_wide
            
        except Exception as e:
            print(f"Error fetching returns from WRDS: {e}")
            raise
    
    def fetch_fundamentals(self, date: datetime) -> pd.DataFrame:
        """
        Fetch fundamental data for style factors from Compustat Global
        """
        if not self.stocks:
            raise ValueError("Universe not built.")
        
        if self.conn is None:
            self.connect()
        
        print(f"Fetching fundamentals from WRDS Compustat Global...")
        
        gvkeys_tuple = tuple(self.stocks)
        
        # Query Compustat Global fundamentals (annual data)
        # Get most recent annual data for each company
        query = """
        SELECT 
            f.gvkey,
            f.sale as revenue,
            f.ni as net_income,
            f.at as total_assets,
            f.ceq as common_equity,
            COALESCE(f.dltt, 0) + COALESCE(f.dlc, 0) as total_debt,
            f.prc as price_close,
            f.cshr as shares_outstanding
        FROM comp.g_funda f
        WHERE f.gvkey IN %s
        AND f.datadate = (SELECT MAX(datadate) FROM comp.g_funda WHERE gvkey = f.gvkey)
        AND f.indfmt = 'INDL'  -- Industrial format
        AND f.datafmt = 'STD'  -- Standard format
        ORDER BY f.gvkey
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=(gvkeys_tuple,))
            
            if df.empty:
                print("No fundamental data found, using defaults...")
                return pd.DataFrame(index=self.stocks)
            
            # Calculate ratios
            df['roe'] = df['net_income'] / df['common_equity']
            df['debt_equity'] = df['total_debt'] / df['common_equity']
            df['profit_margin'] = df['net_income'] / df['revenue']
            df['pe_ratio'] = (df['price_close'] * df['shares_outstanding']) / df['net_income']
            df['pb_ratio'] = (df['price_close'] * df['shares_outstanding']) / df['common_equity']
            
            # Clean up
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.set_index('gvkey')
            
            print(f"Fetched fundamentals for {len(df)} stocks")
            return df
            
        except Exception as e:
            print(f"Warning: Error fetching fundamentals: {e}")
            print("Continuing without fundamental data...")
            return pd.DataFrame(index=self.stocks)
    
    def get_universe_summary(self) -> pd.DataFrame:
        """Get summary of universe"""
        if not self.stocks:
            raise ValueError("Universe not built.")
        return pd.DataFrame(self.stock_data).T.reset_index()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("WRDS connection closed")


class WRDSMacroFactorLoader:
    """Loads macro factors - simplified version"""
    
    def __init__(self):
        self.factor_data = {}
    
    def connect(self):
        """No connection needed for this simplified version"""
        pass
    
    def load_all_factors(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Create placeholder macro factors
        Can be expanded to query actual WRDS macro databases
        """
        print("Creating placeholder macro factors...")
        
        # Create date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create simple factor DataFrame (will be expanded with real data)
        factors_df = pd.DataFrame(index=dates)
        
        # For now, just create dummy factors
        # In production, query WRDS macro data tables
        
        print(f"Loaded macro factors over {len(factors_df)} days")
        return factors_df
    
    def close(self):
        """No cleanup needed"""
        pass


if __name__ == "__main__":
    # Test connection
    print("Testing WRDS connection...")
    universe = WRDSEquityUniverse(top_n=10)
    try:
        universe.connect()
        df = universe.build_universe()
        print(f"\nSuccessfully fetched {len(df)} stocks!")
        print(df)
    finally:
        universe.close()
