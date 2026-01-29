"""
Setup WRDS configuration
Creates .pgpass file for automated authentication
"""
import os
from pathlib import Path

# WRDS credentials
WRDS_USERNAME = "mattyyychan"
WRDS_PASSWORD = "Mc2003uk!!!!"

# Create .pgpass file for WRDS authentication
pgpass_path = Path.home() / ".pgpass"

pgpass_content = f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{WRDS_USERNAME}:{WRDS_PASSWORD}\n"

try:
    with open(pgpass_path, 'w') as f:
        f.write(pgpass_content)
    
    # Set permissions (on Unix-like systems)
    if os.name != 'nt':  # Not Windows
        os.chmod(pgpass_path, 0o600)
    
    print("=" * 60)
    print("WRDS Configuration Complete!")
    print("=" * 60)
    print(f"\nCreated .pgpass file at: {pgpass_path}")
    print(f"Username: {WRDS_USERNAME}")
    print("\nYou can now use WRDS without entering credentials each time.")
    print("\nNext steps:")
    print("  1. Run: python run_dashboard.py")
    print("  2. Click 'Run Model Pipeline' in the dashboard")
    
except Exception as e:
    print(f"Error creating .pgpass file: {e}")
    print("\nAlternative: WRDS will prompt for credentials when needed")
