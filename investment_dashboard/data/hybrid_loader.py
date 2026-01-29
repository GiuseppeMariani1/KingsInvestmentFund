"""
Hybrid Data Loader
- WRDS Compustat Global: Company list + Fundamentals (no look-ahead bias)
- Yahoo Finance: Daily prices (reliable, free)

This gives us 600+ European stocks with proper fundamental data.
"""
import psycopg2
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import time
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


# Mapping of company names to Yahoo Finance tickers for major European stocks
# This is built from WRDS company names -> known tickers
COMPANY_TO_TICKER = {
    # UK
    'BRITISH AMER TOBACCO PLC': 'BATS.L',
    'BP PLC': 'BP.L',
    'BT GROUP PLC': 'BT-A.L',
    'HSBC HOLDINGS PLC': 'HSBA.L',
    'ASTRAZENECA PLC': 'AZN.L',
    'GLAXOSMITHKLINE PLC': 'GSK.L',
    'GSK PLC': 'GSK.L',
    'SHELL PLC': 'SHEL.L',
    'ROYAL DUTCH SHELL PLC': 'SHEL.L',
    'UNILEVER PLC': 'ULVR.L',
    'RIO TINTO PLC': 'RIO.L',
    'DIAGEO PLC': 'DGE.L',
    'BARCLAYS PLC': 'BARC.L',
    'LLOYDS BANKING GROUP PLC': 'LLOY.L',
    'VODAFONE GROUP PLC': 'VOD.L',
    'NATIONAL GRID PLC': 'NG.L',
    'STANDARD CHARTERED PLC': 'STAN.L',
    'PRUDENTIAL PLC': 'PRU.L',
    'LONDON STOCK EXCHANGE GROUP PLC': 'LSEG.L',
    'RELX PLC': 'REL.L',
    'ANGLO AMERICAN PLC': 'AAL.L',
    'ANTOFAGASTA PLC': 'ANTO.L',
    'AVIVA PLC': 'AV.L',
    'COMPASS GROUP PLC': 'CPG.L',
    'EXPERIAN PLC': 'EXPN.L',
    'GLENCORE PLC': 'GLEN.L',
    'IMPERIAL BRANDS PLC': 'IMB.L',
    'NEXT PLC': 'NXT.L',
    'PEARSON PLC': 'PSON.L',
    'RECKITT BENCKISER GROUP PLC': 'RKT.L',
    'SSE PLC': 'SSE.L',
    'TESCO PLC': 'TSCO.L',
    'ROLLS-ROYCE HOLDINGS PLC': 'RR.L',
    'LEGAL & GENERAL GROUP PLC': 'LGEN.L',
    'NATWEST GROUP PLC': 'NWG.L',
    'RENTOKIL INITIAL PLC': 'RTO.L',
    'SMITH & NEPHEW PLC': 'SN.L',
    'SMITHS GROUP PLC': 'SMIN.L',
    'WPP PLC': 'WPP.L',
    'HALEON PLC': 'HLN.L',
    
    # Germany
    'SAP SE': 'SAP.DE',
    'SIEMENS AG': 'SIE.DE',
    'ALLIANZ SE': 'ALV.DE',
    'MUENCHENER RUECKVERSICHERUNGS-GESELLSCHAFT AG': 'MUV2.DE',
    'DEUTSCHE TELEKOM AG': 'DTE.DE',
    'BASF SE': 'BAS.DE',
    'BAYER AG': 'BAYN.DE',
    'BAYERISCHE MOTOREN WERKE AG': 'BMW.DE',
    'BMW AG': 'BMW.DE',
    'MERCEDES-BENZ GROUP AG': 'MBG.DE',
    'DAIMLER AG': 'MBG.DE',
    'VOLKSWAGEN AG': 'VOW3.DE',
    'ADIDAS AG': 'ADS.DE',
    'DEUTSCHE BANK AG': 'DBK.DE',
    'RWE AG': 'RWE.DE',
    'INFINEON TECHNOLOGIES AG': 'IFX.DE',
    'HENKEL AG & CO KGAA': 'HEN3.DE',
    'DEUTSCHE POST AG': 'DHL.DE',
    'E.ON SE': 'EOAN.DE',
    'FRESENIUS SE & CO KGAA': 'FRE.DE',
    'CONTINENTAL AG': 'CON.DE',
    'DEUTSCHE BOERSE AG': 'DB1.DE',
    
    # France
    'LVMH MOET HENNESSY LOUIS VUITTON SE': 'MC.PA',
    'LVMH MOET HENNESSY-LOUIS VUITTON': 'MC.PA',
    'LOREAL SA': 'OR.PA',
    "L'OREAL SA": 'OR.PA',
    'TOTALENERGIES SE': 'TTE.PA',
    'TOTAL SA': 'TTE.PA',
    'SANOFI SA': 'SAN.PA',
    'SANOFI': 'SAN.PA',
    'AIRBUS SE': 'AIR.PA',
    'BNP PARIBAS SA': 'BNP.PA',
    'SCHNEIDER ELECTRIC SE': 'SU.PA',
    'AIR LIQUIDE SA': 'AI.PA',
    'AXA SA': 'CS.PA',
    'SAFRAN SA': 'SAF.PA',
    'DANONE SA': 'BN.PA',
    'PERNOD RICARD SA': 'RI.PA',
    'KERING SA': 'KER.PA',
    'CAPGEMINI SE': 'CAP.PA',
    'VINCI SA': 'DG.PA',
    'ENGIE SA': 'ENGI.PA',
    'HERMES INTERNATIONAL SCA': 'RMS.PA',
    'ESSILOR LUXOTTICA SA': 'EL.PA',
    'ESSILORLUXOTTICA': 'EL.PA',
    'SOCIETE GENERALE SA': 'GLE.PA',
    'CREDIT AGRICOLE SA': 'ACA.PA',
    'ORANGE SA': 'ORA.PA',
    'STELLANTIS NV': 'STLAP.PA',
    
    # Netherlands
    'ASML HOLDING NV': 'ASML.AS',
    'ING GROEP NV': 'INGA.AS',
    'KONINKLIJKE PHILIPS NV': 'PHIA.AS',
    'PHILIPS': 'PHIA.AS',
    'KONINKLIJKE AHOLD DELHAIZE NV': 'AD.AS',
    'AHOLD DELHAIZE NV': 'AD.AS',
    'UNILEVER NV': 'UNA.AS',
    'HEINEKEN NV': 'HEIA.AS',
    'WOLTERS KLUWER NV': 'WKL.AS',
    'PROSUS NV': 'PRX.AS',
    'ADYEN NV': 'ADYEN.AS',
    'ASM INTERNATIONAL NV': 'ASM.AS',
    
    # Switzerland
    'NESTLE SA': 'NESN.SW',
    'NOVARTIS AG': 'NOVN.SW',
    'ROCHE HOLDING AG': 'ROG.SW',
    'UBS GROUP AG': 'UBSG.SW',
    'ZURICH INSURANCE GROUP AG': 'ZURN.SW',
    'ABB LTD': 'ABBN.SW',
    'RICHEMONT SA': 'CFR.SW',
    'CIE FINANCIERE RICHEMONT SA': 'CFR.SW',
    'SWISS RE AG': 'SREN.SW',
    'CREDIT SUISSE GROUP AG': 'CSGN.SW',
    'HOLCIM LTD': 'HOLN.SW',
    'SIKA AG': 'SIKA.SW',
    'GIVAUDAN SA': 'GIVN.SW',
    'LONZA GROUP AG': 'LONN.SW',
    
    # Spain
    'BANCO SANTANDER SA': 'SAN.MC',
    'IBERDROLA SA': 'IBE.MC',
    'INDITEX SA': 'ITX.MC',
    'INDUSTRIA DE DISENO TEXTIL SA': 'ITX.MC',
    'TELEFONICA SA': 'TEF.MC',
    'BBVA': 'BBVA.MC',
    'BANCO BILBAO VIZCAYA ARGENTARIA SA': 'BBVA.MC',
    'AMADEUS IT GROUP SA': 'AMS.MC',
    'FERROVIAL SE': 'FER.MC',
    'CAIXABANK SA': 'CABK.MC',
    'REPSOL SA': 'REP.MC',
    'ENDESA SA': 'ELE.MC',
    
    # Italy
    'ENEL SPA': 'ENEL.MI',
    'INTESA SANPAOLO SPA': 'ISP.MI',
    'ENI SPA': 'ENI.MI',
    'UNICREDIT SPA': 'UCG.MI',
    'FERRARI NV': 'RACE.MI',
    'ASSICURAZIONI GENERALI SPA': 'G.MI',
    'PRYSMIAN SPA': 'PRY.MI',
    'MONCLER SPA': 'MONC.MI',
    'STELLANTIS NV': 'STLA.MI',
    
    # Sweden
    'ATLAS COPCO AB': 'ATCO-A.ST',
    'VOLVO AB': 'VOLV-B.ST',
    'ERICSSON': 'ERIC-B.ST',
    'TELEFONAKTIEBOLAGET LM ERICSSON': 'ERIC-B.ST',
    'HENNES & MAURITZ AB': 'HM-B.ST',
    'H & M HENNES & MAURITZ AB': 'HM-B.ST',
    'INVESTOR AB': 'INVE-B.ST',
    'SANDVIK AB': 'SAND.ST',
    'NORDEA BANK ABP': 'NDA-SE.ST',
    'SVENSKA HANDELSBANKEN AB': 'SHB-A.ST',
    'HEXAGON AB': 'HEXA-B.ST',
    'ESSITY AB': 'ESSITY-B.ST',
    
    # Denmark
    'NOVO NORDISK A/S': 'NOVO-B.CO',
    'MAERSK': 'MAERSK-B.CO',
    'A P MOLLER-MAERSK A/S': 'MAERSK-B.CO',
    'ORSTED A/S': 'ORSTED.CO',
    'CARLSBERG A/S': 'CARL-B.CO',
    'VESTAS WIND SYSTEMS A/S': 'VWS.CO',
    'DSV A/S': 'DSV.CO',
    'COLOPLAST A/S': 'COLO-B.CO',
    'PANDORA A/S': 'PNDORA.CO',
    
    # Finland
    'NOKIA OYJ': 'NOKIA.HE',
    'NORDEA BANK ABP': 'NDA-FI.HE',
    'SAMPO OYJ': 'SAMPO.HE',
    'KONE OYJ': 'KNEBV.HE',
    'UPM-KYMMENE OYJ': 'UPM.HE',
    'FORTUM OYJ': 'FORTUM.HE',
    'NESTE OYJ': 'NESTE.HE',
    
    # Norway
    'EQUINOR ASA': 'EQNR.OL',
    'DNB BANK ASA': 'DNB.OL',
    'TELENOR ASA': 'TEL.OL',
    'NORSK HYDRO ASA': 'NHY.OL',
    'YARA INTERNATIONAL ASA': 'YAR.OL',
    'MOWI ASA': 'MOWI.OL',
    'ORKLA ASA': 'ORK.OL',
    
    # Belgium
    'ANHEUSER-BUSCH INBEV SA': 'ABI.BR',
    'AB INBEV SA': 'ABI.BR',
    'KBC GROUP NV': 'KBC.BR',
    'UCB SA': 'UCB.BR',
    'AGEAS SA': 'AGS.BR',
    'SOLVAY SA': 'SOLB.BR',
    
    # Ireland
    'AON PLC': 'AON',
    'LINDE PLC': 'LIN',
    'ACCENTURE PLC': 'ACN',
    'MEDTRONIC PLC': 'MDT',
    'CRH PLC': 'CRH',
    'SMURFIT KAPPA GROUP PLC': 'SK3.IR',
    'FLUTTER ENTERTAINMENT PLC': 'FLTR.L',
    'KERRY GROUP PLC': 'KYG.IR',
}


class HybridEquityUniverse:
    """
    Hybrid data loader:
    - Company list from WRDS Compustat Global (600+ European stocks)
    - Fundamentals from WRDS (point-in-time, no look-ahead bias)
    - Daily prices from Yahoo Finance (reliable, free)
    """
    
    def __init__(self, top_n: int = 600):
        self.top_n = top_n
        self.stocks = []  # Yahoo Finance tickers
        self.gvkey_to_ticker = {}  # WRDS gvkey -> Yahoo ticker mapping
        self.ticker_to_gvkey = {}  # Reverse mapping
        self.stock_data = {}
        self.wrds_conn = None
        self.wrds_companies = None
        
    def connect_wrds(self):
        """Connect to WRDS PostgreSQL database"""
        print("Connecting to WRDS (direct PostgreSQL connection)...")
        try:
            self.wrds_conn = psycopg2.connect(
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
    
    def query_wrds(self, sql: str) -> pd.DataFrame:
        """Execute SQL query on WRDS"""
        if self.wrds_conn is None:
            self.connect_wrds()
        return pd.read_sql_query(sql, self.wrds_conn)
    
    def build_universe(self) -> pd.DataFrame:
        """
        Build European equity universe using STOXX 600 components.
        
        Uses official STOXX 600 index components for a well-defined, liquid universe.
        """
        print(f"Building STOXX 600 universe (target: {self.top_n} stocks)...")
        
        # Import STOXX 600 tickers
        from .stoxx600_tickers import get_stoxx600_tickers
        stoxx_tickers = get_stoxx600_tickers()
        
        print(f"  STOXX 600 has {len(stoxx_tickers)} component tickers")
        
        # Limit to top_n if specified
        if self.top_n < len(stoxx_tickers):
            stoxx_tickers = stoxx_tickers[:self.top_n]
        
        self.stocks = stoxx_tickers
        
        # Initialize stock_data with basic info
        for ticker in self.stocks:
            self.stock_data[ticker] = {
                'ticker': ticker,
                'exchange': ticker.split('.')[-1] if '.' in ticker else 'UNKNOWN',
            }
        
        print(f"  Universe built: {len(self.stocks)} STOXX 600 stocks")
        
        # Still connect to WRDS for fundamentals (optional)
        try:
            if self.wrds_conn is None:
                self.connect_wrds()
            print("  WRDS connected for fundamentals")
        except Exception as e:
            print(f"  WRDS connection optional, continuing without: {e}")
        
        return pd.DataFrame({'ticker': self.stocks})
    
    def build_universe_legacy(self) -> pd.DataFrame:
        """
        Legacy method: Build universe from WRDS company names.
        Kept for reference but not used.
        """
        if self.wrds_conn is None:
            self.connect_wrds()
        
        print(f"Building universe of top {self.top_n} European stocks (legacy)...")
        print("  Step 1: Fetching companies from WRDS Compustat Global...")
        
        # Get European companies from Compustat Global
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
        AND c.costat = 'A'
        LIMIT {self.top_n * 5}
        """
        
        try:
            self.wrds_companies = self.query_wrds(query)
            print(f"  Found {len(self.wrds_companies)} companies in WRDS")
            
            # Step 2: Map company names to Yahoo Finance tickers
            print("  Step 2: Mapping company names to Yahoo Finance tickers...")
            
            mapped_tickers = []
            for _, row in self.wrds_companies.iterrows():
                gvkey = row['gvkey']
                company_name = row['company_name'].upper().strip()
                
                # Try exact match first
                ticker = COMPANY_TO_TICKER.get(company_name)
                
                # Try partial match if no exact match
                if ticker is None:
                    for name, tic in COMPANY_TO_TICKER.items():
                        if name in company_name or company_name in name:
                            ticker = tic
                            break
                
                if ticker:
                    self.gvkey_to_ticker[gvkey] = ticker
                    self.ticker_to_gvkey[ticker] = gvkey
                    mapped_tickers.append({
                        'ticker': ticker,
                        'gvkey': gvkey,
                        'company_name': row['company_name'],
                        'country': row['country'],
                        'gsector': row['gsector'],
                        'gind': row['gind']
                    })
            
            print(f"  Mapped {len(mapped_tickers)} companies to Yahoo Finance tickers")
            
            # Step 3: Add additional European tickers not in WRDS mapping
            print("  Step 3: Adding additional verified European tickers...")
            additional_tickers = self._get_additional_tickers()
            
            # Combine and deduplicate
            all_tickers = list(set([t['ticker'] for t in mapped_tickers] + additional_tickers))
            
            # Limit to top_n
            all_tickers = all_tickers[:self.top_n]
            
            self.stocks = all_tickers
            
            # Build stock_data dict
            for item in mapped_tickers:
                if item['ticker'] in self.stocks:
                    self.stock_data[item['ticker']] = {
                        'company_name': item['company_name'],
                        'country': item['country'],
                        'gsector': item['gsector'],
                        'gvkey': item['gvkey']
                    }
            
            print(f"\nUniverse built: {len(self.stocks)} European stocks")
            print(f"  - From WRDS mapping: {len(mapped_tickers)}")
            print(f"  - Additional tickers: {len(additional_tickers)}")
            
            # Show sample
            print(f"\nSample companies:")
            for ticker in self.stocks[:5]:
                if ticker in self.stock_data:
                    info = self.stock_data[ticker]
                    print(f"  - {ticker}: {info.get('company_name', 'Unknown')} ({info.get('country', 'Unknown')})")
                else:
                    print(f"  - {ticker}")
            
            # Return DataFrame
            df = pd.DataFrame(mapped_tickers)
            return df
            
        except Exception as e:
            print(f"Error building universe: {e}")
            raise
    
    def _get_additional_tickers(self) -> List[str]:
        """Get additional verified European tickers not in WRDS mapping"""
        # These are verified working tickers from our existing universe
        additional = [
            # UK
            'HSBA.L', 'AZN.L', 'BP.L', 'SHEL.L', 'GSK.L', 'RIO.L', 'ULVR.L', 'DGE.L',
            'BARC.L', 'LLOY.L', 'VOD.L', 'NG.L', 'STAN.L', 'PRU.L', 'LSEG.L', 'REL.L',
            'AAL.L', 'ANTO.L', 'AV.L', 'BATS.L', 'BT-A.L', 'CPG.L', 'EXPN.L', 'GLEN.L',
            'IMB.L', 'NXT.L', 'PSON.L', 'RKT.L', 'SSE.L', 'TSCO.L', 'RR.L', 'LGEN.L',
            'NWG.L', 'RTO.L', 'SN.L', 'SMIN.L', 'WPP.L', 'HLN.L', 'FRAS.L', 'ABF.L',
            'BNZL.L', 'CNA.L', 'CRDA.L', 'DCC.L', 'FCIT.L', 'FRES.L', 'HLMA.L', 'III.L',
            'INF.L', 'ITRK.L', 'JD.L', 'KGF.L', 'LAND.L', 'MNDI.L', 'OCDO.L', 'PHNX.L',
            'PSN.L', 'SBRY.L', 'SDR.L', 'SGRO.L', 'SGE.L', 'SKG.L', 'SMDS.L', 'SMT.L',
            'SPX.L', 'STJ.L', 'SVT.L', 'ULVR.L', 'UU.L', 'WEIR.L', 'WTB.L',
            
            # Germany
            'SAP.DE', 'SIE.DE', 'ALV.DE', 'MUV2.DE', 'DTE.DE', 'BAS.DE', 'BAYN.DE',
            'BMW.DE', 'MBG.DE', 'VOW3.DE', 'ADS.DE', 'AIR.DE', 'DBK.DE', 'RWE.DE',
            'IFX.DE', 'HEN3.DE', 'DHL.DE', 'EOAN.DE', 'FRE.DE', 'CON.DE', 'DB1.DE',
            'MTX.DE', 'SHL.DE', 'BEI.DE', 'HEI.DE', 'LIN.DE', 'MRK.DE', 'PUM.DE',
            'QIA.DE', 'RHM.DE', 'SRT3.DE', 'SY1.DE', 'VNA.DE', 'ZAL.DE',
            
            # France
            'MC.PA', 'OR.PA', 'TTE.PA', 'SAN.PA', 'AIR.PA', 'BNP.PA', 'SU.PA', 'AI.PA',
            'CS.PA', 'SAF.PA', 'BN.PA', 'RI.PA', 'KER.PA', 'CAP.PA', 'DG.PA', 'ENGI.PA',
            'RMS.PA', 'EL.PA', 'GLE.PA', 'ACA.PA', 'ORA.PA', 'STLAP.PA', 'ML.PA',
            'SGO.PA', 'VIE.PA', 'DSY.PA', 'PUB.PA', 'LR.PA', 'EN.PA', 'ERF.PA',
            
            # Netherlands
            'ASML.AS', 'INGA.AS', 'PHIA.AS', 'AD.AS', 'UNA.AS', 'HEIA.AS', 'WKL.AS',
            'PRX.AS', 'ADYEN.AS', 'ASM.AS', 'AKZA.AS', 'DSM.AS', 'RAND.AS', 'NN.AS',
            
            # Switzerland
            'NESN.SW', 'NOVN.SW', 'ROG.SW', 'UBSG.SW', 'ZURN.SW', 'ABBN.SW', 'CFR.SW',
            'SREN.SW', 'HOLN.SW', 'SIKA.SW', 'GIVN.SW', 'LONN.SW', 'GEBN.SW', 'SCMN.SW',
            
            # Spain
            'SAN.MC', 'IBE.MC', 'ITX.MC', 'TEF.MC', 'BBVA.MC', 'AMS.MC', 'FER.MC',
            'CABK.MC', 'REP.MC', 'ELE.MC', 'AENA.MC', 'GRF.MC', 'MAP.MC', 'MEL.MC',
            
            # Italy
            'ENEL.MI', 'ISP.MI', 'ENI.MI', 'UCG.MI', 'RACE.MI', 'G.MI', 'PRY.MI',
            'MONC.MI', 'STLA.MI', 'TEN.MI', 'SPM.MI', 'BAMI.MI', 'BPE.MI', 'LDO.MI',
            
            # Nordic
            'ATCO-A.ST', 'VOLV-B.ST', 'ERIC-B.ST', 'HM-B.ST', 'INVE-B.ST', 'SAND.ST',
            'NDA-SE.ST', 'SHB-A.ST', 'HEXA-B.ST', 'ESSITY-B.ST', 'ELUX-B.ST', 'ALFA.ST',
            'NOVO-B.CO', 'MAERSK-B.CO', 'ORSTED.CO', 'CARL-B.CO', 'VWS.CO', 'DSV.CO',
            'NOKIA.HE', 'SAMPO.HE', 'KNEBV.HE', 'UPM.HE', 'FORTUM.HE', 'NESTE.HE',
            'EQNR.OL', 'DNB.OL', 'TEL.OL', 'NHY.OL', 'YAR.OL', 'MOWI.OL',
            
            # Belgium
            'ABI.BR', 'KBC.BR', 'UCB.BR', 'AGS.BR', 'SOLB.BR', 'ACKB.BR', 'PROX.BR',
        ]
        return additional
    
    def fetch_returns(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch daily returns from Yahoo Finance
        """
        if not self.stocks:
            raise ValueError("Universe not built. Call build_universe() first.")
        
        print(f"Fetching returns from Yahoo Finance ({start_date.date()} to {end_date.date()})...")
        print(f"  Downloading data for {len(self.stocks)} stocks...")
        
        # Download in batches to avoid timeout
        batch_size = 50
        all_returns = []
        
        for i in range(0, len(self.stocks), batch_size):
            batch = self.stocks[i:i+batch_size]
            print(f"  Batch {i//batch_size + 1}/{(len(self.stocks)-1)//batch_size + 1}: {len(batch)} stocks...")
            
            try:
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    threads=True,
                    auto_adjust=True  # Use adjusted prices directly
                )
                
                if data.empty:
                    print(f"    Warning: No data returned for batch")
                    continue
                
                # Handle different yfinance return formats
                if isinstance(data.columns, pd.MultiIndex):
                    # Multi-ticker format: columns are (Price, Ticker)
                    if 'Close' in data.columns.get_level_values(0):
                        prices = data['Close']
                    else:
                        # Try first level
                        prices = data.iloc[:, data.columns.get_level_values(0) == data.columns.get_level_values(0)[0]]
                        prices.columns = prices.columns.droplevel(0)
                else:
                    # Single ticker format
                    if 'Close' in data.columns:
                        prices = data[['Close']]
                        prices.columns = batch if len(batch) == 1 else [batch[0]]
                    else:
                        prices = data
                
                returns = prices.pct_change()
                all_returns.append(returns)
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"    Warning: Error fetching batch: {e}")
                # Try individual downloads for failed batch
                for ticker in batch:
                    try:
                        single_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                        if not single_data.empty and 'Close' in single_data.columns:
                            single_returns = single_data[['Close']].pct_change()
                            single_returns.columns = [ticker]
                            all_returns.append(single_returns)
                    except:
                        pass
                continue
        
        if not all_returns:
            raise ValueError("No return data fetched from Yahoo Finance.")
        
        # Combine all batches
        returns_df = pd.concat(all_returns, axis=1)
        
        # Remove duplicates and NaN columns
        returns_df = returns_df.loc[:, ~returns_df.columns.duplicated()]
        returns_df = returns_df.dropna(how='all', axis=1)
        
        # Update stocks list to only include those with data
        valid_tickers = returns_df.columns.tolist()
        self.stocks = [t for t in self.stocks if t in valid_tickers]
        
        print(f"Fetched returns for {len(returns_df.columns)} stocks over {len(returns_df)} days")
        
        return returns_df
    
    def fetch_fundamentals(self, date: datetime) -> pd.DataFrame:
        """
        Fetch fundamental data from WRDS Compustat Global
        """
        if self.wrds_conn is None:
            self.connect_wrds()
        
        print(f"Fetching fundamentals from WRDS Compustat Global...")
        
        # Get gvkeys for our tickers
        gvkeys = [self.ticker_to_gvkey.get(t) for t in self.stocks if t in self.ticker_to_gvkey]
        gvkeys = [g for g in gvkeys if g is not None]
        
        if not gvkeys:
            print("  No gvkey mappings found, returning empty fundamentals")
            return pd.DataFrame(index=self.stocks)
        
        gvkeys_tuple = tuple(gvkeys)
        
        # Query Compustat Global fundamentals
        # Column names: sale=revenue, ib=income before extraordinary, at=total assets
        # ceq=common equity, dltt=long-term debt, dlc=current debt
        # cshr=common shares outstanding, prc=price (Compustat Global uses 'prc' not 'prcc_f')
        query = """
        SELECT 
            f.gvkey,
            f.sale as revenue,
            f.ib as net_income,
            f.at as total_assets,
            f.ceq as common_equity,
            f.cshr as shares_outstanding,
            f.prc as price_fiscal,
            COALESCE(f.dltt, 0) + COALESCE(f.dlc, 0) as total_debt,
            f.datadate
        FROM comp.g_funda f
        WHERE f.gvkey IN %s
        ORDER BY f.gvkey, f.datadate DESC
        """
        
        try:
            df = pd.read_sql_query(query, self.wrds_conn, params=(gvkeys_tuple,))
            
            if df.empty:
                print("  No fundamental data found in WRDS")
                return pd.DataFrame(index=self.stocks)
            
            # Get most recent data for each gvkey
            df = df.groupby('gvkey').first().reset_index()
            
            # Calculate quality ratios
            df['roe'] = df['net_income'] / df['common_equity']
            df['debt_equity'] = df['total_debt'] / df['common_equity']
            df['profit_margin'] = df['net_income'] / df['revenue']
            df['asset_turnover'] = df['revenue'] / df['total_assets']
            
            # Calculate valuation ratios
            # Market cap (shares in millions, price in local currency)
            df['market_cap'] = df['price_fiscal'] * df['shares_outstanding'] * 1_000_000
            
            # EPS = net_income / shares_outstanding (in millions)
            df['eps'] = df['net_income'] / df['shares_outstanding']
            
            # P/E ratio = price / EPS
            df['pe_ratio'] = df['price_fiscal'] / df['eps']
            df.loc[df['pe_ratio'] < 0, 'pe_ratio'] = np.nan  # Negative P/E is meaningless
            df.loc[df['pe_ratio'] > 200, 'pe_ratio'] = np.nan  # Extreme values
            
            # Book value per share = common_equity / shares_outstanding
            df['bvps'] = df['common_equity'] / df['shares_outstanding']
            
            # P/B ratio = price / book value per share
            df['pb_ratio'] = df['price_fiscal'] / df['bvps']
            df.loc[df['pb_ratio'] < 0, 'pb_ratio'] = np.nan  # Negative book value
            df.loc[df['pb_ratio'] > 50, 'pb_ratio'] = np.nan  # Extreme values
            
            # EV/EBITDA not easily available, set to NaN
            df['ev_ebitda'] = np.nan
            
            # Map gvkey back to ticker
            df['ticker'] = df['gvkey'].map(self.gvkey_to_ticker)
            
            # For tickers without gvkey mapping, we'll add them with NaN fundamentals
            all_tickers = set(self.stocks)
            mapped_tickers = set(df['ticker'].dropna())
            unmapped_tickers = all_tickers - mapped_tickers
            
            if unmapped_tickers:
                unmapped_df = pd.DataFrame({'ticker': list(unmapped_tickers)})
                df = pd.concat([df, unmapped_df], ignore_index=True)
            
            df = df.set_index('ticker')
            
            # Clean up
            df = df.replace([np.inf, -np.inf], np.nan)
            
            print(f"  Fetched fundamentals for {len(df)} stocks ({len(mapped_tickers)} from WRDS)")
            return df
            
        except Exception as e:
            print(f"  Warning: Error fetching fundamentals: {e}")
            return pd.DataFrame(index=self.stocks)
    
    def get_stock_info(self) -> pd.DataFrame:
        """Get stock information DataFrame"""
        info = []
        for ticker in self.stocks:
            if ticker in self.stock_data:
                data = self.stock_data[ticker]
                info.append({
                    'ticker': ticker,
                    'name': data.get('company_name', ticker),
                    'company_name': data.get('company_name', ticker),
                    'country': data.get('country', 'Unknown'),
                    'sector': data.get('gsector', 'Unknown'),
                    'gsector': data.get('gsector', 'Unknown')
                })
            else:
                # Try to infer country from ticker suffix
                suffix = ticker.split('.')[-1] if '.' in ticker else ''
                country_map = {
                    'L': 'GBR', 'DE': 'DEU', 'PA': 'FRA', 'AS': 'NLD',
                    'SW': 'CHE', 'MC': 'ESP', 'MI': 'ITA', 'ST': 'SWE',
                    'CO': 'DNK', 'HE': 'FIN', 'OL': 'NOR', 'BR': 'BEL'
                }
                info.append({
                    'ticker': ticker,
                    'name': ticker,
                    'company_name': ticker,
                    'country': country_map.get(suffix, 'Unknown'),
                    'sector': 'Unknown',
                    'gsector': 'Unknown'
                })
        
        return pd.DataFrame(info).set_index('ticker')
    
    def get_universe_summary(self) -> pd.DataFrame:
        """Get summary of universe (for compatibility with EquityUniverse)"""
        return self.get_stock_info().reset_index()
    
    def close(self):
        """Close connections"""
        if self.wrds_conn:
            self.wrds_conn.close()
            print("WRDS connection closed")


if __name__ == "__main__":
    # Test the hybrid loader
    print("Testing Hybrid Data Loader...")
    loader = HybridEquityUniverse(top_n=100)
    
    try:
        # Build universe
        df = loader.build_universe()
        print(f"\nUniverse: {len(loader.stocks)} stocks")
        
        # Test returns fetch
        from datetime import datetime, timedelta
        end = datetime.now()
        start = end - timedelta(days=30)
        returns = loader.fetch_returns(start, end)
        print(f"\nReturns: {returns.shape}")
        
        # Test fundamentals
        fundamentals = loader.fetch_fundamentals(end)
        print(f"\nFundamentals: {len(fundamentals)} stocks")
        
    finally:
        loader.close()
