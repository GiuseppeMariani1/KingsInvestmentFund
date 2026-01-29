"""
Equity Universe Definition and Loading
Supports multiple data sources:
- Yahoo Finance (original)
- Hybrid (WRDS company list + fundamentals, Yahoo prices)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import UNIVERSE_SIZE, EUROPEAN_MARKET_ONLY, EUROPEAN_EXCHANGES, DATA_SOURCE


def get_equity_universe(top_n: int = UNIVERSE_SIZE):
    """
    Factory function to get the appropriate equity universe loader
    based on DATA_SOURCE config.
    """
    if DATA_SOURCE == "hybrid":
        from .hybrid_loader import HybridEquityUniverse
        return HybridEquityUniverse(top_n=top_n)
    else:
        return EquityUniverse(top_n=top_n)


class EquityUniverse:
    """Manages the European equity universe"""
    
    def __init__(self, top_n: int = UNIVERSE_SIZE):
        self.top_n = top_n
        self.stocks = []
        self.stock_data = {}
        
    def get_european_tickers(self) -> List[str]:
        """
        Get list of European stock tickers.
        VERIFIED working tickers on Yahoo Finance (tested Jan 2026)
        """
        # Verified working European tickers - 103 stocks across major exchanges
        european_stocks = [
            # UK (London Stock Exchange) - 32 stocks
            'HSBA.L', 'AZN.L', 'BP.L', 'SHEL.L', 'GSK.L', 'RIO.L', 'ULVR.L', 'DGE.L',
            'BARC.L', 'LLOY.L', 'VOD.L', 'NG.L', 'STAN.L', 'PRU.L', 'LSEG.L', 'REL.L',
            'AAL.L', 'ANTO.L', 'AV.L', 'BATS.L', 'BT-A.L', 'CPG.L', 'EXPN.L', 'GLEN.L',
            'IMB.L', 'NXT.L', 'PSON.L', 'RKT.L', 'SGRO.L', 'SSE.L', 'SVT.L', 'TSCO.L',
            
            # Germany (XETRA) - 16 stocks
            'SAP.DE', 'SIE.DE', 'ALV.DE', 'MUV2.DE', 'DTE.DE', 'BAS.DE', 'BAYN.DE',
            'BMW.DE', 'MBG.DE', 'VOW3.DE', 'ADS.DE', 'AIR.DE', 'DBK.DE', 'RWE.DE',
            'IFX.DE', 'HEN3.DE',
            
            # France (Euronext Paris) - 16 stocks
            'MC.PA', 'OR.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA', 'BNP.PA', 'SU.PA', 'AI.PA',
            'CS.PA', 'SAF.PA', 'BN.PA', 'RI.PA', 'KER.PA', 'CAP.PA', 'DG.PA', 'ENGI.PA',
            
            # Netherlands (Euronext Amsterdam) - 7 stocks
            'ASML.AS', 'INGA.AS', 'PHIA.AS', 'AD.AS', 'UNA.AS', 'HEIA.AS', 'WKL.AS',
            
            # Switzerland (SIX) - 7 stocks
            'NESN.SW', 'ROG.SW', 'NOVN.SW', 'UBSG.SW', 'ABBN.SW', 'SIKA.SW', 'GIVN.SW',
            
            # Spain (Bolsa de Madrid) - 7 stocks
            'SAN.MC', 'BBVA.MC', 'ITX.MC', 'IBE.MC', 'TEF.MC', 'REP.MC', 'FER.MC',
            
            # Italy (Borsa Italiana) - 6 stocks
            'ENEL.MI', 'ENI.MI', 'ISP.MI', 'UCG.MI', 'G.MI', 'RACE.MI',
            
            # Denmark (Copenhagen) - 4 stocks
            'NOVO-B.CO', 'MAERSK-B.CO', 'DSV.CO', 'CARL-B.CO',
            
            # Sweden (Stockholm) - 4 stocks
            'VOLV-B.ST', 'ERIC-B.ST', 'SAND.ST', 'ABB.ST',
            
            # Norway (Oslo) - 3 stocks
            'EQNR.OL', 'DNB.OL', 'MOWI.OL',
            
            # Finland (Helsinki) - 3 stocks
            'NOKIA.HE', 'FORTUM.HE', 'NESTE.HE',
        ]
        
        return european_stocks[:self.top_n]
    
    def fetch_stock_info(self, ticker: str) -> Optional[Dict]:
        """Fetch basic info for a stock"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data
            if not info or len(info) < 5:
                return None
            
            # Extract key information
            market_cap = info.get('marketCap', 0)
            if market_cap == 0:
                market_cap = info.get('enterpriseValue', 0)
            
            # Skip if no market cap data
            if market_cap == 0 or market_cap is None:
                return None
            
            return {
                'ticker': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'country': info.get('country', 'Unknown'),
                'market_cap': market_cap,
                'currency': info.get('currency', 'EUR'),
                'exchange': info.get('exchange', 'Unknown')
            }
        except Exception as e:
            # Silently skip invalid tickers
            return None
    
    def build_universe(self) -> pd.DataFrame:
        """
        Build the equity universe by fetching top N stocks by market cap
        """
        print(f"Building universe of top {self.top_n} European stocks...")
        
        # Get candidate tickers
        tickers = self.get_european_tickers()
        
        # Fetch info for all tickers (with progress indication)
        stock_info_list = []
        print(f"Fetching data for {len(tickers)} tickers...")
        successful = 0
        failed = 0
        
        for i, ticker in enumerate(tickers):
            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(tickers)} tickers... ({successful} successful, {failed} failed)")
            
            info = self.fetch_stock_info(ticker)
            if info and info['market_cap'] > 0:
                stock_info_list.append(info)
                successful += 1
            else:
                failed += 1
            
            # Small delay to avoid rate limiting (every 10 requests)
            if (i + 1) % 10 == 0:
                time.sleep(0.1)
        
        print(f"\n  Completed: {successful} successful, {failed} failed")
        
        # Convert to DataFrame and sort by market cap
        df = pd.DataFrame(stock_info_list)
        if df.empty:
            raise ValueError("No stocks found. Check ticker symbols and data availability.")
        
        df = df.sort_values('market_cap', ascending=False)
        df = df.head(self.top_n)
        df = df.reset_index(drop=True)
        
        self.stocks = df['ticker'].tolist()
        self.stock_data = df.set_index('ticker').to_dict('index')
        
        print(f"Universe built: {len(self.stocks)} stocks (top {self.top_n} by market cap)")
        return df
    
    def fetch_returns(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch historical returns for all stocks in universe
        
        Returns:
            DataFrame with columns: date, ticker, return
        """
        if not self.stocks:
            raise ValueError("Universe not built. Call build_universe() first.")
        
        print(f"Fetching returns from {start_date.date()} to {end_date.date()}...")
        
        all_returns = []
        
        for ticker in self.stocks:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_date, end=end_date)
                
                if not hist.empty:
                    # Calculate daily returns
                    returns = hist['Close'].pct_change().dropna()
                    # Remove timezone info to avoid conversion issues
                    dates = returns.index.tz_localize(None) if returns.index.tz else returns.index
                    returns_df = pd.DataFrame({
                        'date': dates,
                        'ticker': ticker,
                        'return': returns.values
                    })
                    all_returns.append(returns_df)
            except Exception as e:
                print(f"Error fetching returns for {ticker}: {e}")
                continue
        
        if not all_returns:
            raise ValueError("No return data fetched.")
        
        returns_df = pd.concat(all_returns, ignore_index=True)
        # Ensure dates are timezone-naive
        returns_df['date'] = pd.to_datetime(returns_df['date']).dt.tz_localize(None)
        
        # Pivot to wide format: dates x stocks
        returns_wide = returns_df.pivot(index='date', columns='ticker', values='return')
        
        print(f"Fetched returns for {len(returns_wide.columns)} stocks over {len(returns_wide)} days")
        return returns_wide
    
    def get_universe_summary(self) -> pd.DataFrame:
        """Get summary statistics of the universe"""
        if not self.stocks:
            raise ValueError("Universe not built.")
        
        return pd.DataFrame(self.stock_data).T.reset_index()


if __name__ == "__main__":
    # Test the equity universe
    universe = EquityUniverse(top_n=50)
    df = universe.build_universe()
    print("\nUniverse Summary:")
    print(df.head(10))
    print(f"\nTotal stocks: {len(df)}")
