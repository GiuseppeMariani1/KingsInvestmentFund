"""
Discover available WRDS databases and tables
"""
import psycopg2
import pandas as pd

# Credentials
USERNAME = "mattyyychan"
PASSWORD = "Mc2003uk!!!!"

print("=" * 70)
print("WRDS Database Discovery")
print("=" * 70)

try:
    # Connect
    print("\nConnecting to WRDS...")
    conn = psycopg2.connect(
        host='wrds-pgdata.wharton.upenn.edu',
        port=9737,
        database='wrds',
        user=USERNAME,
        password=PASSWORD
    )
    print("Connected successfully!")
    
    # List all schemas (databases)
    print("\n" + "=" * 70)
    print("Available WRDS Libraries/Schemas:")
    print("=" * 70)
    
    schema_query = """
    SELECT schema_name 
    FROM information_schema.schemata 
    WHERE schema_name NOT LIKE 'pg_%' 
    AND schema_name != 'information_schema'
    ORDER BY schema_name
    """
    
    schemas = pd.read_sql_query(schema_query, conn)
    
    # Look for relevant schemas
    relevant_keywords = ['comp', 'crsp', 'global', 'europe', 'worldscope', 'datastream', 
                         'thomson', 'funda', 'security']
    
    print("\nAll available schemas:")
    for schema in schemas['schema_name']:
        print(f"  - {schema}")
    
    print("\n" + "=" * 70)
    print("Potentially Relevant Schemas for European Stocks:")
    print("=" * 70)
    
    for schema in schemas['schema_name']:
        schema_lower = schema.lower()
        if any(keyword in schema_lower for keyword in relevant_keywords):
            print(f"\n📊 {schema}")
            
            # Try to get table count
            try:
                table_query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema}'
                LIMIT 10
                """
                tables = pd.read_sql_query(table_query, conn)
                if not tables.empty:
                    print(f"   Sample tables: {', '.join(tables['table_name'].head(5).tolist())}")
            except:
                pass
    
    # Try to find European stock data
    print("\n" + "=" * 70)
    print("Searching for European Stock Data...")
    print("=" * 70)
    
    # Common possible schemas
    possible_schemas = [
        'comp',  # Compustat
        'compm',  # Compustat Monthly
        'compd',  # Compustat Daily  
        'comp_global',  # Compustat Global
        'crsp',  # CRSP
        'crsp_a_stock',  # CRSP Annual Stock
        'worldscope',  # Worldscope (Thomson Reuters)
        'datastream',  # Datastream
        'tr_ds_equities',  # Thomson Reuters Datastream Equities
    ]
    
    for schema in possible_schemas:
        if schema in schemas['schema_name'].values:
            print(f"\n✓ Found: {schema}")
            try:
                # List tables
                table_query = f"""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = '{schema}'
                ORDER BY table_name
                LIMIT 20
                """
                tables = pd.read_sql_query(table_query, conn)
                print(f"  Tables: {', '.join(tables['table_name'].tolist())}")
            except Exception as e:
                print(f"  Error listing tables: {e}")
    
    conn.close()
    print("\n" + "=" * 70)
    print("Discovery Complete!")
    print("=" * 70)
    print("\nNext step: Based on available databases, we'll use the best source for European stocks.")
    
except Exception as e:
    print(f"\nError: {e}")
    print("\nMake sure:")
    print("  1. You're on King's College network/VPN")
    print("  2. Your WRDS account is active")
    print("  3. Username/password are correct")
