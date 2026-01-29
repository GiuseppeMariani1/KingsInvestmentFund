# Quick Start Guide - King's Investment Fund Dashboard

## Step-by-Step Instructions

### Step 1: Navigate to Project Directory

Open PowerShell or Command Prompt and navigate to the KIF directory:

```powershell
cd "C:\Users\mattc\Downloads\Finance Analytics Programming\Finance Analytics\KIF"
```

### Step 2: Install Dependencies

Install all required Python packages (including WRDS):

```powershell
pip install -r requirements.txt
```

**Expected output**: Packages will be installed (streamlit, pandas, numpy, yfinance, plotly, scikit-learn, scipy, wrds, psycopg2-binary)

**Note**: The WRDS library requires Python 3.7+ and PostgreSQL drivers.

**Note**: If you encounter permission errors, use:
```powershell
pip install --user -r requirements.txt
```

### Step 3: Verify WRDS Connection (First Time Only)

Test your WRDS connection:

```powershell
python -c "import wrds; db = wrds.Connection(username='mattyyychan'); print('✅ Connected to WRDS!'); db.close()"
```

If prompted, follow the instructions to set up `.pgpass` file for authentication.

### Step 4: Run the Dashboard

You have **two options**:

#### Option A: Using the Convenience Script (Easiest)

```powershell
python run_dashboard.py
```

#### Option B: Using Streamlit Directly

```powershell
streamlit run investment_dashboard/dashboard/app.py
```

### Step 5: Access the Dashboard

After running Step 4, you should see:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

1. **Open your web browser** (Chrome, Firefox, Edge, etc.)
2. **Navigate to**: `http://localhost:8501`
3. The dashboard should open automatically

### Step 6: Run the Model Pipeline

Once the dashboard loads:

1. **Look at the sidebar** on the left
2. **Click the button**: "🚀 Run Model Pipeline"
3. **Wait for processing** (this may take 4-8 minutes):
   - Connecting to WRDS
   - Loading European stock universe from Compustat Global
   - Fetching stock returns (3 years)
   - Loading fundamentals for style factors
   - Fitting PLS model (3-year lookback)
   - Fitting Elastic Net models (1-year lookback)
   - Calculating expected returns
   - Building portfolio

**Progress indicators**:
- You'll see progress messages in the terminal/console
- The dashboard will show a spinner while processing
- When complete, you'll see: "✅ Model fitted successfully!"

### Step 7: Explore the Dashboard

After the model finishes, navigate through the pages:

- **Overview**: Model summary, key metrics, expected return distribution
- **Factor Analysis**: PLS component loadings, time series, factor importance
- **Stock Rankings**: Top-ranked stocks by expected return
- **Portfolio**: Current holdings, weights, sector breakdown
- **Performance**: (Placeholder for future backtesting)

---

## Troubleshooting

### Issue: "Module not found" error

**Solution**: Make sure you installed dependencies:
```powershell
pip install -r requirements.txt
```

### Issue: "Port 8501 already in use"

**Solution**: Streamlit will automatically use the next available port (8502, 8503, etc.). Check the terminal output for the correct URL.

### Issue: "WRDS authentication failed"

**Solution**: 
- Verify credentials in `wrds_credentials.py`
- Ensure you're on King's College network or VPN
- Check your WRDS account is active
- Try running: `python -c "import wrds; wrds.Connection(username='mattyyychan')"`

### Issue: "Insufficient data" error

**Solution**:
- Ensure you have internet connection
- Wait a few minutes and try again (Yahoo Finance rate limits)
- Check that the date range is valid (3 years of history required)

### Issue: Dashboard won't open in browser

**Solution**:
- Copy the URL from the terminal (e.g., `http://localhost:8501`)
- Paste it manually into your browser
- Make sure no firewall is blocking the connection

---

## Expected Runtime (with WRDS)

- **WRDS Connection**: 5-10 seconds
- **Data Loading**: 30-60 seconds (much faster than Yahoo Finance!)
- **PLS Model Fitting**: 1-2 minutes
- **Elastic Net Fitting**: 2-5 minutes (one model per stock)
- **Total**: 4-8 minutes for first run

**Note**: WRDS is much faster and more reliable than Yahoo Finance.

---

## What Happens During Model Execution

1. **Data Loading** (from WRDS):
   - Connects to WRDS database
   - Fetches top 500 European stocks from Compustat Global
   - Loads fundamentals (ROE, debt/equity, profit margins, etc.)
   - Calculates style factors (value, momentum, quality, volatility)

2. **PLS Model**:
   - Compresses factors into return-driving components
   - Identifies what drives European equity returns

3. **Elastic Net**:
   - Evaluates each stock's exposure to return drivers
   - Shrinks weak relationships to zero
   - Calculates model-implied expected returns

4. **Portfolio Construction**:
   - Ranks stocks by expected return
   - Selects top 30 stocks
   - Constructs equal-weighted portfolio

---

## Next Steps

After running the model:

1. **Review the Overview** page to see model metrics
2. **Check Factor Analysis** to understand what drives returns
3. **Examine Stock Rankings** to see top picks
4. **View Portfolio** to see the constructed holdings
5. **Adjust parameters** in `investment_dashboard/utils/config.py` if needed

---

## Command Summary

```powershell
# Navigate to project
cd "C:\Users\mattc\Downloads\Finance Analytics Programming\Finance Analytics\KIF"

# Install dependencies (one-time)
pip install -r requirements.txt

# Run dashboard (choose one)
python run_dashboard.py
# OR
streamlit run investment_dashboard/dashboard/app.py
```

---

## Need Help?

- Check `README.md` for more details
- Review `MODEL_FRAMEWORK.md` for strategy explanation
- See `STRATEGY_ALIGNMENT.md` for implementation details
- Check `IMPLEMENTATION_SUMMARY.md` for technical details
