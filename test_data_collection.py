#!/usr/bin/env python3
"""
Test script for the new data collection functionality
"""

import sys
import os
import pandas as pd
import sqlite3
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_collection():
    """Test the data collection script"""
    logger.info("🧪 Testing data collection script...")
    
    try:
        # Import the collector
        sys.path.append('src/data_ingestion')
        from collect_sample_data import HospitalDataCollector
        
        # Initialize collector
        collector = HospitalDataCollector()
        
        # Test with a smaller date range for faster testing
        start_date = "2024-01-01"
        end_date = "2024-03-31"  # 3 months of data
        
        logger.info(f"📊 Generating test data from {start_date} to {end_date}...")
        
        # Collect data
        admission_df, capacity_df = collector.collect_all_data(
            start_date=start_date,
            end_date=end_date,
            include_covid_api=False  # Skip API for testing
        )
        
        # Verify data
        logger.info("🔍 Verifying generated data...")
        
        # Check admission data
        assert len(admission_df) > 0, "No admission data generated"
        assert 'date' in admission_df.columns, "Missing date column"
        assert 'location' in admission_df.columns, "Missing location column"
        assert 'admission_count' in admission_df.columns, "Missing admission_count column"
        assert 'disease_type' in admission_df.columns, "Missing disease_type column"
        assert 'age_group' in admission_df.columns, "Missing age_group column"
        
        # Check capacity data
        assert len(capacity_df) > 0, "No capacity data generated"
        assert 'location' in capacity_df.columns, "Missing location column"
        assert 'total_beds' in capacity_df.columns, "Missing total_beds column"
        assert 'icu_beds' in capacity_df.columns, "Missing icu_beds column"
        assert 'available_beds' in capacity_df.columns, "Missing available_beds column"
        
        # Check database
        db_path = "data/hospital_forecasting.db"
        assert os.path.exists(db_path), f"Database not created: {db_path}"
        
        conn = sqlite3.connect(db_path)
        
        # Check tables exist
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        assert 'hospital_admissions' in tables['name'].values, "hospital_admissions table missing"
        assert 'hospital_capacity' in tables['name'].values, "hospital_capacity table missing"
        
        # Check data in tables
        admission_count = pd.read_sql_query("SELECT COUNT(*) as count FROM hospital_admissions", conn)
        capacity_count = pd.read_sql_query("SELECT COUNT(*) as count FROM hospital_capacity", conn)
        
        assert admission_count['count'].iloc[0] > 0, "No data in hospital_admissions table"
        assert capacity_count['count'].iloc[0] > 0, "No data in hospital_capacity table"
        
        conn.close()
        
        # Check CSV files
        csv_files = [
            "data/raw/hospital_admissions.csv",
            "data/raw/hospital_capacity.csv",
            "data/raw/combined_hospital_data.csv"
        ]
        
        for csv_file in csv_files:
            assert os.path.exists(csv_file), f"CSV file not created: {csv_file}"
            df = pd.read_csv(csv_file)
            assert len(df) > 0, f"Empty CSV file: {csv_file}"
        
        # Check summary report
        summary_file = "reports/data_collection_summary.json"
        assert os.path.exists(summary_file), f"Summary file not created: {summary_file}"
        
        logger.info("✅ All tests passed!")
        
        # Print summary
        logger.info(f"\n📊 Test Results:")
        logger.info(f"- Admission records: {len(admission_df):,}")
        logger.info(f"- Capacity records: {len(capacity_df):,}")
        logger.info(f"- Locations: {admission_df['location'].nunique()}")
        logger.info(f"- Diseases: {admission_df['disease_type'].nunique()}")
        logger.info(f"- Date range: {admission_df['date'].min()} to {admission_df['date'].max()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False

def test_pipeline_integration():
    """Test that the collected data works with the existing pipeline"""
    logger.info("🔗 Testing pipeline integration...")
    
    try:
        # Check if processed data can be created
        if os.path.exists("data/processed/combined_forecast_data.csv"):
            logger.info("✅ Processed data already exists")
            return True
        
        # Try to run preprocessing
        logger.info("🔄 Running data preprocessing...")
        import subprocess
        result = subprocess.run([
            sys.executable, "src/data_processing/preprocess_data.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ Data preprocessing completed successfully")
            return True
        else:
            logger.warning(f"⚠️ Data preprocessing failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Pipeline integration test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    logger.info("🏥 Testing Hospital Data Collection")
    logger.info("=" * 40)
    
    tests = [
        ("Data Collection", test_data_collection),
        ("Pipeline Integration", test_pipeline_integration)
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
        logger.info("🎉 All tests passed! Data collection is working correctly.")
        logger.info("\n📋 Next steps:")
        logger.info("1. python src/data_processing/preprocess_data.py")
        logger.info("2. python src/models/prophet_forecasting.py")
        logger.info("3. streamlit run src/visualization/streamlit_dashboard.py")
    else:
        logger.warning(f"⚠️ {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main() 