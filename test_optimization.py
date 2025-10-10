#!/usr/bin/env python3
"""
Simple test for hyperparameter optimization
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

def test_data_loading():
    """Test if we can load the data"""
    print("🔍 Testing data loading...")
    
    data_path = "data/processed/combined_forecast_data.csv"
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"Locations: {df['location'].unique()}")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_prophet_import():
    """Test if Prophet can be imported"""
    print("🔍 Testing Prophet import...")
    
    try:
        from prophet import Prophet
        print("✅ Prophet imported successfully")
        
        # Test basic model creation
        model = Prophet()
        print("✅ Prophet model created successfully")
        return True
    except Exception as e:
        print(f"❌ Error importing Prophet: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing hyperparameter optimization setup...")
    
    # Test data loading
    if not test_data_loading():
        return
    
    # Test Prophet import
    if not test_prophet_import():
        return
    
    print("🎉 All tests passed! Ready for optimization.")

if __name__ == "__main__":
    main()
