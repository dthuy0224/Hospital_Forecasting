#!/usr/bin/env python3
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/hospital_forecasting.db')

# Check tables
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print("Tables:", tables['name'].tolist())

# Check data counts
admission_count = pd.read_sql_query("SELECT COUNT(*) as count FROM hospital_admissions", conn)
capacity_count = pd.read_sql_query("SELECT COUNT(*) as count FROM hospital_capacity", conn)

print("Admissions count:", admission_count['count'].iloc[0])
print("Capacity count:", capacity_count['count'].iloc[0])

# Check sample data
if admission_count['count'].iloc[0] > 0:
    sample_admissions = pd.read_sql_query("SELECT * FROM hospital_admissions LIMIT 5", conn)
    print("\nSample admissions:")
    print(sample_admissions)

if capacity_count['count'].iloc[0] > 0:
    sample_capacity = pd.read_sql_query("SELECT * FROM hospital_capacity LIMIT 5", conn)
    print("\nSample capacity:")
    print(sample_capacity)

conn.close() 