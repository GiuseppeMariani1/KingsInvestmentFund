"""
Check Thomson Reuters Worldscope tables
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

print("Checking tr_worldscope tables...")
tables = pd.read_sql_query("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'tr_worldscope' 
    ORDER BY table_name
""", conn)

print(f"\nFound {len(tables)} tables in tr_worldscope:")
print(tables.to_string())

# Check for key tables
print("\n\nLooking for equity/price/fundamental tables...")
for table in tables['table_name']:
    if any(keyword in table.lower() for keyword in ['equity', 'price', 'fundamental', 'company', 'security']):
        print(f"  -> {table}")

conn.close()
