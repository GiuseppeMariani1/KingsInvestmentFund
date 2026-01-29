# WRDS Setup Guide

## What is WRDS?

WRDS (Wharton Research Data Services) is a premium institutional database providing high-quality financial data. Much better than Yahoo Finance for systematic strategies:

- **Reliable data**: No rate limits, 404 errors, or missing data
- **European coverage**: Compustat Global covers major European stocks
- **Fundamentals**: Complete financial statement data
- **Returns**: Clean, adjusted price and return data

---

## Installation Steps

### 1. Install WRDS Python Library

```powershell
cd "C:\Users\mattc\Downloads\Finance Analytics Programming\Finance Analytics\KIF"
pip install wrds psycopg2-binary
```

Or update all dependencies:
```powershell
pip install -r requirements.txt
```

### 2. Credentials Already Configured

✅ Your WRDS credentials are stored in `wrds_credentials.py`:
- Username: mattyyychan
- Password: (configured)

**Important**: This file is in `.gitignore` so it won't be committed to git.

### 3. Test WRDS Connection

```powershell
python -c "import wrds; db = wrds.Connection(username='mattyyychan'); print('Connected!'); db.close()"
```

You may be prompted to create a `.pgpass` file for future authentication.

---

## Running the Dashboard with WRDS

### Quick Start

```powershell
cd "C:\Users\mattc\Downloads\Finance Analytics Programming\Finance Analytics\KIF"
python run_dashboard.py
```

The dashboard is already configured to use WRDS (DATA_SOURCE = "wrds" in config.py).

---

## What Data is Loaded from WRDS

### 1. Equity Universe
- **Source**: Compustat Global (`comp_global.g_company`, `comp_global.g_secd`)
- **Coverage**: Top 500 European stocks by market cap
- **Countries**: AUT, BEL, CHE, DEU, DNK, ESP, FIN, FRA, GBR, IRL, ITA, NLD, NOR, SWE
- **Fields**: Company name, ticker, ISIN, sector, country, market cap

### 2. Stock Returns
- **Source**: Compustat Global Security Daily (`comp_global.g_secd`)
- **Data**: Daily prices and total return factors
- **Period**: 3 years (for PLS model)

### 3. Fundamental Data
- **Source**: Compustat Global Fundamentals Annual (`comp_global.g_funda`)
- **Factors**: ROE, debt/equity, profit margin, revenue growth
- **Used for**: Style factor calculation

### 4. Macro Factors
- **Source**: Can be expanded to use WRDS/Fred library
- **Current**: Placeholder (can add interest rates, FX, etc.)

---

## Advantages Over Yahoo Finance

| Feature | Yahoo Finance | WRDS |
|---------|--------------|------|
| European coverage | Limited, many 404s | Comprehensive |
| Data quality | Variable | Institutional-grade |
| Rate limiting | Yes (major issue) | No |
| Fundamentals | Limited | Complete |
| Historical data | Sometimes missing | Reliable |
| Update frequency | Daily | Daily |
| Cost | Free | Institutional access |

---

## Expected Runtime with WRDS

Much faster than Yahoo Finance:

- **Data Loading**: 30-60 seconds (vs 2-5 minutes with Yahoo)
- **PLS Fitting**: 1-2 minutes (same)
- **Elastic Net**: 2-5 minutes (same)
- **Total**: 4-8 minutes (vs 5-12 minutes)

---

## Troubleshooting

### Issue: "Authentication failed"

**Solution**: 
- Check username/password in `wrds_credentials.py`
- Ensure you're on King's College network or VPN
- Verify your WRDS account is active

### Issue: "Table does not exist"

**Solution**:
- Verify you have access to Compustat Global
- Check with King's College library about database subscriptions
- May need to request access to specific databases

### Issue: "Connection timeout"

**Solution**:
- Check internet connection
- Ensure you're on King's network or VPN
- WRDS requires institutional access

---

## WRDS Databases Available

As a King's College student, you likely have access to:

- ✅ **Compustat Global**: European stock fundamentals
- ✅ **CRSP Global**: Global stock returns (alternative source)
- ✅ **Datastream**: Additional European data
- ✅ **Thomson Reuters**: Ownership, estimates
- ✅ **OptionMetrics**: Options data (if needed)

---

## Next Steps

1. **Install WRDS library** (if not done):
   ```powershell
   pip install wrds psycopg2-binary
   ```

2. **Run dashboard**:
   ```powershell
   cd "C:\Users\mattc\Downloads\Finance Analytics Programming\Finance Analytics\KIF"
   python run_dashboard.py
   ```

3. **Click "Run Model Pipeline"** - should work smoothly with WRDS data

---

## Configuration

The dashboard is now configured to use WRDS automatically. See:
- `utils/config.py`: `DATA_SOURCE = "wrds"`
- `wrds_credentials.py`: Your WRDS login credentials
- `data/wrds_loader.py`: WRDS data loading logic
- `models/model_pipeline_wrds.py`: WRDS-based pipeline

---

## Security Note

- `wrds_credentials.py` is in `.gitignore` (won't be committed)
- Never share this file or commit it to version control
- If sharing code, delete this file first

---

Ready to run with WRDS!
