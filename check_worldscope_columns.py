"""
Check columns in Worldscope tables
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

print("Checking wrds_ws_company columns...")
query1 = """
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_schema = 'tr_worldscope' AND table_name = 'wrds_ws_company'
ORDER BY ordinal_position
"""
cols1 = pd.read_sql_query(query1, conn)
print(cols1.to_string())

print("\n\nSample data from wrds_ws_company:")
sample1 = pd.read_sql_query("SELECT * FROM tr_worldscope.wrds_ws_company LIMIT 5", conn)
print(sample1.to_string())

print("\n\nChecking wrds_ws_funda columns...")
query2 = """
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_schema = 'tr_worldscope' AND table_name = 'wrds_ws_funda'
ORDER BY ordinal_position
LIMIT 30
"""
cols2 = pd.read_sql_query(query2, conn)
print(cols2.to_string())

conn.close()
