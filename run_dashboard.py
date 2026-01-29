"""
Simple script to run the King's Investment Fund Dashboard
"""
import subprocess
import sys
import os

# Change to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Run streamlit
subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "investment_dashboard/dashboard/app.py"
])
