"""
Thomson Reuters Worldscope Data Loader for WRDS
Uses tr_worldscope schema for European stock data
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
    WRDS_USERNAME = "mattyyychan"
    WRDS_PASSWORD = "Mc2003uk!!!!"


class WorldscopeEquityUniverse:
    """
    European equity universe using Thomson Reuters Worldscope
    
    Key Worldscope Item Codes:
    - item1000: Company Name
    - item6001: Country Code
    - item7001: SIC Code
    - item01001: Total Revenue
    - item01051: Net Income
    - item02999: Total Assets
    - item03501: Common Equity
    - item03351: Total Debt
    - item05301: Price (Close)
    - item08001: Market Value
    """
    
    def __init__(self, top_n: int = 500):
        self.top_n = top_n
        self.stocks = []
        self.stock_data = {}
        self.conn = None
        
    def connect(self):
        """Connect to WRDS PostgreSQL"""
        print("Connecting to WRDS...")
        try:
            self.conn = psycopg2.connect(
                host='wrds-pgdata.wharton.upenn.edu',
                port=9737,
                database='wrds',
                user=WRDS_USERNAME,
                password=WRDS_PASSWORD
            )
            print("Connected to WRDS successfully!")
        except Exception as e:
            print(f"Error connecting: {e}")
            raise
    
    def query(self, sql: str, params=None) -> pd.DataFrame:
        """Execute SQL query"""
        if self.conn is None:
            self.connect()
        return pd.read_sql_query(sql, self.conn, params=params)
    
    def build_universe(self) -> pd.DataFrame:
        """
        Build European equity universe from Worldscope
        """
        if self.conn is None:
            self.connect()
        
        print(f"Building universe from Worldscope...")
        
        # Get European stocks with latest market cap from fundamentals
        # item08001 = Market Value
        # item1000 = Company Name
        # item6001 = Country
        query = f"""
        SELECT 
            f.code,
            c.item1000 as company_name,
            c.item6001 as country,
            f.item08001 as market_cap,
            c.isin
        FROM tr_worldscope.wrds_ws_funda f
        LEFT JOIN tr_worldscope.wrds_ws_company c ON f.code = c.code
        WHERE c.item6001 IN ('AUSTRIA', 'BELGIUM', 'SWITZERLAND', 'GERMANY', 'DENMARK', 
                             'SPAIN', 'FINLAND', 'FRANCE', 'UNITED KINGDOM', 'IRELAND', 
                             'ITALY', 'NETHERLANDS', 'NORWAY', 'SWEDEN')
        AND f.item08001 IS NOT NULL
        AND f.item08001 > 0
        AND f.freq = 'A'
        ORDER BY f.item08001 DESC
        LIMIT {self.top_n}
        """
        
        try:
            df = self.query(query)
            
            if df.empty:
                raise ValueError("No European stocks found in Worldscope")
            
            # Clean up
            df = df.dropna(subset=['code'])
            df['code'] = df['code'].astype(int)
            
            self.stocks = df['code'].tolist()
            self.stock_data = df.set_index('code').to_dict('index')
            
            print(f"Universe built: {len(self.stocks)} European stocks")
            print(f"\nTop 5 by market cap:")
            for i, row in df.head(5).iterrows():
                mkt_cap_b = row['market_cap'] / 1000 if pd.notna(row['market_cap']) else 0
                name = row['company_name'] if pd.notna(row['company_name']) else f"Code {row['code']}"
                country = row['country'] if pd.notna(row['country']) else "Unknown"
                print(f"  {i+1}. {name} ({country}) - ${mkt_cap_b:.1f}B")
            
            return df
            
        except Exception as e:
            print(f"Error building universe: {e}")
            raise
    
    def fetch_returns(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch daily returns from Worldscope daily data
        """
        if not self.stocks:
            raise ValueError("Universe not built.")
        
        if self.conn is None:
            self.connect()
        
        print(f"Fetching returns from Worldscope ({start_date.date()} to {end_date.date()})...")
        
        codes_tuple = tuple(self.stocks)
        
        # Query from wsddata (daily data)
        # item05301 = Price Close
        # item05001 = Total Return Index
        query = """
        SELECT 
            date_ as date,
            code,
            item05301 as price_close,
            item05001 as total_return_index
        FROM tr_worldscope.wsddata
        WHERE code IN %s
        AND date_ >= %s
        AND date_ <= %s
        AND item05301 IS NOT NULL
        ORDER BY code, date_
        """
        
        try:
            df = self.query(query, params=(codes_tuple, start_date, end_date))
            
            if df.empty:
                raise ValueError("No return data fetched")
            
            # Calculate returns
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['code', 'date'])
            
            # Use total return index if available, else price returns
            if 'total_return_index' in df.columns and df['total_return_index'].notna().sum() > 100:
                df['return'] = df.groupby('code')['total_return_index'].pct_change()
            else:
                df['return'] = df.groupby('code')['price_close'].pct_change()
            
            # Pivot to wide format
            returns_wide = df.pivot(index='date', columns='code', values='return')
            returns_wide = returns_wide.dropna(how='all', axis=1)
            
            print(f"Fetched returns for {len(returns_wide.columns)} stocks over {len(returns_wide)} days")
            
            return returns_wide
            
        except Exception as e:
            print(f"Error fetching returns: {e}")
            raise
    
    def fetch_fundamentals(self, date: datetime) -> pd.DataFrame:
        """
        Fetch fundamentals from Worldscope annual data
        """
        if not self.stocks:
            raise ValueError("Universe not built.")
        
        if self.conn is None:
            self.connect()
        
        print(f"Fetching fundamentals from Worldscope...")
        
        codes_tuple = tuple(self.stocks)
        
        # Worldscope item codes for fundamentals
        query = """
        SELECT 
            code,
            item01001 as revenue,
            item01051 as net_income,
            item02999 as total_assets,
            item03501 as common_equity,
            item03351 as total_debt,
            item08001 as market_cap
        FROM tr_worldscope.wrds_ws_funda
        WHERE code IN %s
        AND freq = 'A'
        ORDER BY code
        """
        
        try:
            df = self.query(query, params=(codes_tuple,))
            
            if df.empty:
                print("No fundamental data found")
                return pd.DataFrame(index=self.stocks)
            
            # Calculate ratios
            df['roe'] = df['net_income'] / df['common_equity']
            df['debt_equity'] = df['total_debt'] / df['common_equity']
            df['profit_margin'] = df['net_income'] / df['revenue']
            
            # Clean
            df = df.replace([np.inf, -np.inf], np.nan)
            df['code'] = df['code'].astype(int)
            df = df.set_index('code')
            
            print(f"Fetched fundamentals for {len(df)} stocks")
            return df
            
        except Exception as e:
            print(f"Warning: {e}")
            return pd.DataFrame(index=self.stocks)
    
    def get_universe_summary(self) -> pd.DataFrame:
        """Get universe summary"""
        if not self.stocks:
            raise ValueError("Universe not built.")
        return pd.DataFrame(self.stock_data).T.reset_index()
    
    def close(self):
        """Close connection"""
        if self.conn:
            self.conn.close()
            print("WRDS connection closed")


class WorldscopeMacroLoader:
    """Simple macro loader - placeholder"""
    
    def __init__(self):
        pass
    
    def connect(self):
        pass
    
    def load_all_factors(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create placeholder macro factors"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.DataFrame(index=dates)
    
    def close(self):
        pass


if __name__ == "__main__":
    print("Testing Worldscope loader...")
    universe = WorldscopeEquityUniverse(top_n=20)
    try:
        df = universe.build_universe()
        print(f"\nSuccessfully built universe with {len(df)} stocks!")
    finally:
        universe.close()
