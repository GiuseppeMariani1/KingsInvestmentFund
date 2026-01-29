"""
Simple query to get European stocks
"""
import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host='wrds-pgdata.wharton.upenn.edu',
    port=9737,
    database='wrds',
    user='mattyyychan',
    password='Mc2003uk!!!!'
)

print("Querying European stocks from Worldscope company table...")

# Just get company info first
query = """
SELECT 
    code,
    item1000 as company_name,
    item6001 as country
FROM tr_worldscope.wrds_ws_company
WHERE item6001 IN ('AUSTRIA', 'BELGIUM', 'SWITZERLAND', 'GERMANY', 'DENMARK', 
                   'SPAIN', 'FINLAND', 'FRANCE', 'UNITED KINGDOM', 'IRELAND', 
                   'ITALY', 'NETHERLANDS', 'NORWAY', 'SWEDEN')
AND item1000 IS NOT NULL
LIMIT 50
"""

df = pd.read_sql_query(query, conn)
print(f"\nFound {len(df)} European stocks")
print(df.head(20))

# Now try to get some price data for these stocks
if not df.empty:
    codes = tuple(df['code'].head(5).tolist())
    print(f"\n\nTesting price data for first 5 stocks (codes: {codes})...")
    
    price_query = """
    SELECT code, date_, item05301 as price
    FROM tr_worldscope.wsddata
    WHERE code IN %s
    AND date_ >= '2024-01-01'
    LIMIT 100
    """
    
    prices = pd.read_sql_query(price_query, conn, params=(codes,))
    print(f"Found {len(prices)} price observations")
    if not prices.empty:
        print(prices.head(20))

conn.close()
print("\nDone!")
