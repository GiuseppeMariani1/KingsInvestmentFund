# WRDS Dashboard - Run Instructions

## WRDS Setup Complete!

The .pgpass file has been created successfully. Now run the dashboard:

---

## Steps to Run

### 1. Navigate to KIF folder

```powershell
cd "C:\Users\mattc\Downloads\Finance Analytics Programming\Finance Analytics\KIF"
```

### 2. Run the dashboard

```powershell
python run_dashboard.py
```

The dashboard will open at `http://localhost:8501` (or 8502, 8503, etc. if port is in use)

### 3. In the dashboard

- Click "🚀 Run Model Pipeline" in the sidebar
- Wait for processing (4-8 minutes)

---

## What's Happening

The dashboard now uses WRDS with direct PostgreSQL connection:

1. **Connects to WRDS** - Using your King's College credentials
2. **Queries Compustat Global** - Top 500 European stocks by market cap
3. **Fetches 3 years of returns** - From g_secd table
4. **Loads fundamentals** - ROE, debt ratios, margins from g_funda
5. **Fits PLS model** - Identifies European return drivers
6. **Fits Elastic Net** - Evaluates stock exposures
7. **Builds portfolio** - Top 30 stocks, equal-weighted

---

## Expect to See

In the terminal:
```
Connecting to WRDS (direct PostgreSQL connection)...
Connected to WRDS successfully!
Building universe of top 500 European stocks from WRDS Compustat Global...
Universe built: 500 European stocks

Top 5 by market cap:
  1. Company Name (GBR) - $XXX.XB
  2. Company Name (DEU) - $XXX.XB
  ...
```

---

## If Issues Occur

### Issue: "relation comp_global.g_company does not exist"

**Possible causes:**
- King's College may not have Compustat Global subscription
- Database name might be different

**Solution**: Check which WRDS databases you have access to:
- Log in to https://wrds-www.wharton.upenn.edu/
- Go to "Data" → "Subscriptions"
- Verify "Compustat Global" is listed

### Issue: Connection timeout

**Solution**:
- Ensure you're on King's College network or VPN
- WRDS requires institutional network access

### Alternative: Use Sample Mode

If WRDS isn't working, we can switch back to Yahoo Finance temporarily or use a smaller sample.

---

## Ready to Run!

```powershell
cd "C:\Users\mattc\Downloads\Finance Analytics Programming\Finance Analytics\KIF"
python run_dashboard.py
```

Then click "Run Model Pipeline" in the dashboard sidebar.
