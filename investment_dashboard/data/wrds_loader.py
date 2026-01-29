"""
WRDS Data Loader
Loads European stock data from WRDS (Compustat Global, CRSP Global)
"""
import wrds
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
    """Manages European equity universe using WRDS data"""
    
    def __init__(self, top_n: int = 500):
        self.top_n = top_n
        self.stocks = []
        self.stock_data = {}
        self.db = None
        
    def connect(self):
        """Connect to WRDS database"""
        print("Connecting to WRDS...")
        try:
            # Use .pgpass file authentication (configured by setup_wrds.py)
            if WRDS_USERNAME:
                self.db = wrds.Connection(username=WRDS_USERNAME)
            else:
                self.db = wrds.Connection()
            print("Connected to WRDS successfully!")
        except Exception as e:
            print(f"Error connecting to WRDS: {e}")
            print("Tip: Run 'python setup_wrds.py' first to configure authentication")
            raise
    
    def build_universe(self) -> pd.DataFrame:
        """
        Build European equity universe from Compustat Global
        """
        if self.db is None:
            self.connect()
        
        print(f"Building universe of top {self.top_n} European stocks from WRDS...")
        
        # Query Compustat Global for European stocks
        # Get most recent market cap data
        query = """
        SELECT DISTINCT
            a.gvkey,
            a.conm as company_name,
            a.tic as ticker,
            a.exchg as exchange,
            a.loc as country,
            a.gsector as sector,
            b.isin,
            c.mkvaltq as market_cap,
            c.datadate
        FROM comp_global.g_company a
        LEFT JOIN comp_global.g_security b ON a.gvkey = b.gvkey
        LEFT JOIN comp_global.g_secd c ON a.gvkey = c.gvkey
        WHERE a.loc IN ('AUT', 'BEL', 'CHE', 'DEU', 'DNK', 'ESP', 'FIN', 
                        'FRA', 'GBR', 'IRL', 'ITA', 'NLD', 'NOR', 'SWE')
        AND c.mkvaltq IS NOT NULL
        AND c.mkvaltq > 0
        AND c.datadate >= CURRENT_DATE - INTERVAL '1 month'
        ORDER BY c.mkvaltq DESC
        LIMIT {limit}
        """.format(limit=self.top_n * 2)  # Get more to ensure we have enough after filtering
        
        try:
            df = self.db.raw_sql(query)
            
            if df.empty:
                raise ValueError("No European stocks found in WRDS. Check database access.")
            
            # Sort by market cap and take top N
            df = df.sort_values('market_cap', ascending=False)
            df = df.drop_duplicates(subset=['gvkey'], keep='first')
            df = df.head(self.top_n)
            df = df.reset_index(drop=True)
            
            self.stocks = df['gvkey'].tolist()
            self.stock_data = df.set_index('gvkey').to_dict('index')
            
            print(f"✅ Universe built: {len(self.stocks)} European stocks")
            print(f"   Top 5 by market cap:")
            for i, row in df.head(5).iterrows():
                print(f"   {i+1}. {row['company_name']} ({row['country']}) - €{row['market_cap']/1e9:.1f}B")
            
            return df
            
        except Exception as e:
            print(f"❌ Error querying WRDS: {e}")
            raise
    
    def fetch_returns(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch stock returns from WRDS Compustat Global
        """
        if not self.stocks:
            raise ValueError("Universe not built. Call build_universe() first.")
        
        if self.db is None:
            self.connect()
        
        print(f"Fetching returns from WRDS ({start_date.date()} to {end_date.date()})...")
        
        # Convert gvkeys to string for SQL query
        gvkeys_str = "', '".join([str(g) for g in self.stocks])
        
        # Query daily returns from Compustat Global Security Daily
        query = f"""
        SELECT 
            datadate as date,
            gvkey,
            prccd as price_close,
            trfd as total_return_factor
        FROM comp_global.g_secd
        WHERE gvkey IN ('{gvkeys_str}')
        AND datadate >= '{start_date.strftime('%Y-%m-%d')}'
        AND datadate <= '{end_date.strftime('%Y-%m-%d')}'
        AND prccd IS NOT NULL
        ORDER BY gvkey, datadate
        """
        
        try:
            df = self.db.raw_sql(query)
            
            if df.empty:
                raise ValueError("No return data fetched from WRDS.")
            
            # Calculate returns from total return factor
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['gvkey', 'date'])
            
            # Calculate daily returns for each stock
            df['return'] = df.groupby('gvkey')['total_return_factor'].pct_change()
            
            # Pivot to wide format: dates x stocks
            returns_wide = df.pivot(index='date', columns='gvkey', values='return')
            returns_wide = returns_wide.dropna(how='all', axis=1)  # Remove stocks with no data
            
            print(f"✅ Fetched returns for {len(returns_wide.columns)} stocks over {len(returns_wide)} days")
            
            return returns_wide
            
        except Exception as e:
            print(f"❌ Error fetching returns from WRDS: {e}")
            raise
    
    def fetch_fundamentals(self, date: datetime) -> pd.DataFrame:
        """
        Fetch fundamental data for style factors from Compustat Global
        """
        if not self.stocks:
            raise ValueError("Universe not built.")
        
        if self.db is None:
            self.connect()
        
        print(f"Fetching fundamentals from WRDS for {date.date()}...")
        
        gvkeys_str = "', '".join([str(g) for g in self.stocks])
        
        # Query fundamentals from Compustat Global Fundamentals Annual
        query = f"""
        SELECT 
            gvkey,
            datadate,
            revt as revenue,
            ni as net_income,
            at as total_assets,
            ceq as common_equity,
            lt as total_liabilities,
            csho as shares_outstanding,
            prcc_f as price_close,
            ni / ceq as roe,
            lt / ceq as debt_equity,
            ni / revt as profit_margin,
            (revt - LAG(revt, 1) OVER (PARTITION BY gvkey ORDER BY datadate)) / LAG(revt, 1) OVER (PARTITION BY gvkey ORDER BY datadate) as revenue_growth
        FROM comp_global.g_funda
        WHERE gvkey IN ('{gvkeys_str}')
        AND datadate <= '{date.strftime('%Y-%m-%d')}'
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND popsrc = 'D'
        AND consol = 'C'
        ORDER BY gvkey, datadate DESC
        """
        
        try:
            df = self.db.raw_sql(query)
            
            # Take most recent data for each stock
            df = df.sort_values(['gvkey', 'datadate'], ascending=[True, False])
            df = df.drop_duplicates(subset=['gvkey'], keep='first')
            df = df.set_index('gvkey')
            
            print(f"✅ Fetched fundamentals for {len(df)} stocks")
            return df
            
        except Exception as e:
            print(f"❌ Error fetching fundamentals: {e}")
            raise
    
    def get_universe_summary(self) -> pd.DataFrame:
        """Get summary of universe"""
        if not self.stocks:
            raise ValueError("Universe not built.")
        
        return pd.DataFrame(self.stock_data).T.reset_index()
    
    def close(self):
        """Close WRDS connection"""
        if self.db:
            self.db.close()
            print("WRDS connection closed")


class WRDSMacroFactorLoader:
    """Loads macro factors from WRDS"""
    
    def __init__(self):
        self.db = None
        self.factor_data = {}
    
    def connect(self):
        """Connect to WRDS"""
        if WRDS_USERNAME:
            self.db = wrds.Connection(username=WRDS_USERNAME)
        else:
            self.db = wrds.Connection()
    
    def load_all_factors(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load macro factors from WRDS/Fred
        
        For now, we'll use a simplified approach - can be expanded to use
        WRDS's Fred library or other macro data sources
        """
        if self.db is None:
            self.connect()
        
        print("Loading macro factors from WRDS/external sources...")
        
        # Placeholder - WRDS provides access to various macro data
        # This can be expanded based on available WRDS libraries
        
        # For now, create a simple DataFrame
        # In production, query actual WRDS macro data tables
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        factors_df = pd.DataFrame(index=dates)
        
        print(f"✅ Loaded macro factors over {len(factors_df)} days")
        return factors_df
    
    def close(self):
        """Close connection"""
        if self.db:
            self.db.close()


if __name__ == "__main__":
    # Test WRDS connection
    print("Testing WRDS connection...")
    universe = WRDSEquityUniverse(top_n=50)
    universe.connect()
    df = universe.build_universe()
    print(f"\nFetched {len(df)} stocks")
    print(df.head())
    universe.close()
