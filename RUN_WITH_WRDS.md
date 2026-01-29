# Running the Dashboard with WRDS

## Setup Complete!

WRDS authentication has been configured. Follow these steps:

---

## Step-by-Step Instructions

### 1. Navigate to KIF folder

```powershell
cd "C:\Users\mattc\Downloads\Finance Analytics Programming\Finance Analytics\KIF"
```

### 2. Run the dashboard

```powershell
python run_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

### 3. Click "Run Model Pipeline"

In the dashboard sidebar, click the button "Run Model Pipeline"

### 4. Wait for processing (4-8 minutes)

You'll see progress in the terminal:
- Connecting to WRDS
- Building European stock universe
- Fetching returns
- Fitting PLS model
- Fitting Elastic Net models
- Building portfolio

### 5. Explore results

Navigate through the dashboard pages:
- Overview
- Factor Analysis
- Stock Rankings  
- Portfolio
- Performance

---

## What WRDS Provides

- **Universe**: Top 500 European stocks from Compustat Global
- **Returns**: 3 years of daily returns
- **Fundamentals**: ROE, debt/equity, profit margins, revenue growth
- **Quality**: Institutional-grade data, no rate limits

---

## Credentials

- Username: mattyyychan
- Authentication: `.pgpass` file (already configured)
- No need to enter password each time

---

## If WRDS Prompts for Credentials

If the dashboard prompts for WRDS credentials:

1. Enter username: `mattyyychan`
2. Enter password: `Mc2003uk!!!!`
3. Select "Save credentials" if prompted

The `.pgpass` file should handle this automatically, but WRDS may prompt on first connection.

---

## Command Summary

```powershell
# One-time setup (already done)
python setup_wrds.py

# Run dashboard
cd "C:\Users\mattc\Downloads\Finance Analytics Programming\Finance Analytics\KIF"
python run_dashboard.py
```

---

## Ready to run!

The dashboard is configured to use WRDS. Simply run:

```powershell
python run_dashboard.py
```

Then click "Run Model Pipeline" in the sidebar.
