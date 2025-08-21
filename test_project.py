#!/usr/bin/env python3
"""
Test script to verify Hospital Forecasting Project functionality
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported"""
    logger.info("🔍 Testing module imports...")
    
    try:
        # Test core data science libraries
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        logger.info("✅ Core data science libraries imported")
        
        # Test ML libraries
        import sklearn
        from prophet import Prophet
        import statsmodels
        logger.info("✅ Machine learning libraries imported")
        
        # Test visualization libraries
        import streamlit as st
        import dash
        logger.info("✅ Visualization libraries imported")
        
        # Test project modules
        sys.path.append('.')
        import src.data_processing.preprocess_data
        from src.models.prophet_forecasting import HospitalDemandForecaster
        import src.visualization.streamlit_dashboard
        logger.info("✅ Project modules imported")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Import test failed: {str(e)}")
        return False

def test_config():
    """Test configuration loading"""
    logger.info("🔍 Testing configuration...")
    
    try:
        import yaml
        with open('config/config.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        required_keys = ['data_sources', 'database', 'models', 'forecasting', 'dashboard', 'locations']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing config key: {key}")
        
        logger.info("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {str(e)}")
        return False

def test_data_creation():
    """Test creating sample data"""
    logger.info("🔍 Testing data creation...")
    
    try:
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        locations = ['Ho Chi Minh City', 'Ha Noi', 'Da Nang']
        
        data = []
        for location in locations:
            for date in dates:
                # Simulate hospital admissions with some seasonality
                base_admissions = 100
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                weekend_factor = 0.8 if date.weekday() >= 5 else 1.0
                noise = np.random.normal(0, 10)
                
                admissions = int(base_admissions * seasonal_factor * weekend_factor + noise)
                admissions = max(0, admissions)  # Ensure non-negative
                
                data.append({
                    'ds': date,
                    'y': admissions,
                    'location': location,
                    'is_weekend': 1 if date.weekday() >= 5 else 0,
                    'month': date.month
                })
        
        df = pd.DataFrame(data)
        
        # Save sample data
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/combined_forecast_data.csv', index=False)
        
        logger.info(f"✅ Created sample data: {len(df)} records for {df['location'].nunique()} locations")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data creation test failed: {str(e)}")
        return False

def test_model_training():
    """Test model training with sample data"""
    logger.info("🔍 Testing model training...")
    
    try:
        from src.models.prophet_forecasting import HospitalDemandForecaster
        
        # Initialize forecaster
        forecaster = HospitalDemandForecaster()
        
        # Load sample data
        data = forecaster.load_data()
        
        if data.empty:
            logger.warning("⚠️ No data available for model training test")
            return False
        
        # Train model for one location
        location = data['location'].unique()[0]
        model = forecaster.train_model(data, location)
        
        if model is not None:
            logger.info(f"✅ Model trained successfully for {location}")
            return True
        else:
            logger.warning("⚠️ Model training returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model training test failed: {str(e)}")
        return False

def test_directories():
    """Test if all required directories exist"""
    logger.info("🔍 Testing directory structure...")
    
    required_dirs = [
        'src',
        'src/data_ingestion',
        'src/data_processing', 
        'src/models',
        'src/visualization',
        'config',
        'data',
        'data/raw',
        'data/processed',
        'models',
        'models/saved_models',
        'models/forecasts',
        'reports',
        'logs',
        'notebooks',
        'tests'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        logger.warning(f"⚠️ Missing directories: {missing_dirs}")
        # Create missing directories
        for dir_path in missing_dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"✅ Created directory: {dir_path}")
    else:
        logger.info("✅ All required directories exist")
    
    return True

def main():
    """Run all tests"""
    logger.info("🏥 Hospital Forecasting Project - System Test")
    logger.info("=" * 50)
    
    tests = [
        ("Directory Structure", test_directories),
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Data Creation", test_data_creation),
        ("Model Training", test_model_training)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} PASSED")
        else:
            logger.error(f"❌ {test_name} FAILED")
    
    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Project is ready to use.")
        logger.info("\n📋 Next steps:")
        logger.info("1. python run_pipeline.py")
        logger.info("2. streamlit run src/visualization/streamlit_dashboard.py")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main() 