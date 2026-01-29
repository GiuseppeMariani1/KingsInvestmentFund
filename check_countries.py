"""
Check what country values exist in Worldscope
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

print("Checking unique country values in Worldscope...")

query = """
SELECT DISTINCT item6001 as country, COUNT(*) as count
FROM tr_worldscope.wrds_ws_company
WHERE item6001 IS NOT NULL
GROUP BY item6001
ORDER BY count DESC
LIMIT 50
"""

df = pd.read_sql_query(query, conn)
print(df.to_string())

# Also check a few sample rows
print("\n\nSample rows:")
sample = pd.read_sql_query("SELECT code, item1000, item6001 FROM tr_worldscope.wrds_ws_company LIMIT 20", conn)
print(sample.to_string())

conn.close()
