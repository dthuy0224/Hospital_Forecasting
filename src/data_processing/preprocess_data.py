#!/usr/bin/env python3
"""
Data preprocessing script for Hospital Forecasting Project
Cleans and prepares data for machine learning models
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import yaml
from typing import Dict, List, Tuple
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Class to preprocess hospital admission data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self.load_config(config_path)
        self.db_path = "data/hospital_forecasting.db"
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'forecasting': {'prediction_days': 14},
            'locations': ['Ho Chi Minh City', 'Ha Noi', 'Da Nang']
        }
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data from database"""
        logger.info("📊 Loading data from database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load admission data
        admission_df = pd.read_sql_query("""
            SELECT * FROM hospital_admissions 
            ORDER BY date, location
        """, conn)
        
        # Load capacity data
        capacity_df = pd.read_sql_query("""
            SELECT * FROM hospital_capacity
            ORDER BY location
        """, conn)
        
        conn.close()
        
        # Convert date columns
        admission_df['date'] = pd.to_datetime(admission_df['date'])
        capacity_df['date'] = pd.to_datetime(capacity_df['date'])
        
        logger.info(f"✅ Loaded {len(admission_df):,} admission records")
        logger.info(f"✅ Loaded {len(capacity_df):,} capacity records")
        
        return admission_df, capacity_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        logger.info("🧹 Cleaning data...")
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove negative admission counts
        df = df[df['admission_count'] >= 0]
        
        # Remove outliers (values > 3 standard deviations) - but be more conservative
        mean_admissions = df['admission_count'].mean()
        std_admissions = df['admission_count'].std()
        threshold = 5 * std_admissions  # Increased from 3 to 5 to be less aggressive
        
        df = df[np.abs(df['admission_count'] - mean_admissions) <= threshold]
        
        # Fill missing values
        df['disease_type'] = df['disease_type'].fillna('Unknown')
        df['age_group'] = df['age_group'].fillna('Unknown')
        
        # Validate date range (remove future dates)
        df = df[df['date'] <= datetime.now()]
        
        final_rows = len(df)
        logger.info(f"✅ Data cleaned: {initial_rows} → {final_rows} rows ({initial_rows - final_rows} removed)")
        
        # If too much data was removed, log a warning
        if final_rows < initial_rows * 0.5:
            logger.warning(f"⚠️ More than 50% of data was removed during cleaning!")
        
        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for modeling"""
        logger.info("⚙️ Engineering features...")
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Seasonal indicators
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_holiday'] = 0  # Can be enhanced with actual holiday data
        
        # Lag features (previous day admissions)
        df_sorted = df.sort_values(['location', 'date'])
        df_sorted['prev_day_admissions'] = df_sorted.groupby('location')['admission_count'].shift(1)
        df_sorted['prev_week_admissions'] = df_sorted.groupby('location')['admission_count'].shift(7)
        
        # Rolling averages
        df_sorted['ma_7_days'] = df_sorted.groupby('location')['admission_count'].rolling(window=7, min_periods=1).mean().reset_index(0, drop=True)
        df_sorted['ma_30_days'] = df_sorted.groupby('location')['admission_count'].rolling(window=30, min_periods=1).mean().reset_index(0, drop=True)
        
        # Disease severity encoding (simple example)
        severity_map = {
            'COVID-19': 3,
            'Pneumonia': 3,
            'Heart Disease': 3,
            'Flu': 2,
            'Diabetes': 2,
            'Others': 1,
            'Unknown': 1
        }
        df_sorted['disease_severity'] = df_sorted['disease_type'].map(severity_map).fillna(1)
        
        logger.info(f"✅ Added {len([col for col in df_sorted.columns if col not in df.columns])} new features")
        
        return df_sorted
    
    def aggregate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data for forecasting"""
        logger.info("📈 Aggregating data for forecasting...")
        
        # Daily aggregation by location
        daily_agg = df.groupby(['date', 'location']).agg({
            'admission_count': 'sum',
            'disease_severity': 'mean',
            'is_weekend': 'first',
            'is_holiday': 'first',
            'year': 'first',
            'month': 'first',
            'day': 'first',
            'day_of_week': 'first',
            'day_of_year': 'first',
            'week_of_year': 'first',
            'quarter': 'first'
        }).reset_index()
        
        # Add population-based normalization (mock data)
        population_map = {
            'Ho Chi Minh City': 9000000,
            'Ha Noi': 8000000,
            'Da Nang': 1200000,
            'Can Tho': 1200000,
            'Hai Phong': 2000000,
            'Nha Trang': 400000,
            'Hue': 400000,
            'Vung Tau': 300000,
            'Bien Hoa': 1000000,
            'Thu Dau Mot': 500000
        }
        
        daily_agg['population'] = daily_agg['location'].map(population_map).fillna(500000)
        daily_agg['admissions_per_100k'] = (daily_agg['admission_count'] / daily_agg['population']) * 100000
        
        logger.info(f"✅ Aggregated to {len(daily_agg)} daily records")
        
        return daily_agg
    
    def prepare_forecast_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare data for different forecasting models"""
        logger.info("🔮 Preparing data for forecasting models...")
        
        forecast_data = {}
        
        # For each location, create Prophet-format data
        for location in df['location'].unique():
            location_data = df[df['location'] == location].copy()
            location_data = location_data.sort_values('date')
            
            # Prophet format (ds, y)
            prophet_data = location_data[['date', 'admission_count']].copy()
            prophet_data.columns = ['ds', 'y']
            
            # Add additional regressors
            prophet_data['is_weekend'] = location_data['is_weekend'].values
            prophet_data['month'] = location_data['month'].values
            prophet_data['disease_severity'] = location_data['disease_severity'].values
            
            # Fill date gaps to create continuous time series
            prophet_data = self.fill_date_gaps(prophet_data)
            
            forecast_data[location] = prophet_data
        
        logger.info(f"✅ Prepared forecast data for {len(forecast_data)} locations")
        
        return forecast_data
    
    def fill_date_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing dates to create continuous time series for Prophet"""
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Get date range
        min_date = df['ds'].min()
        max_date = df['ds'].max()
        
        # Create complete date range
        full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        # Check if gaps exist
        original_days = len(df)
        expected_days = len(full_date_range)
        
        if original_days < expected_days:
            logger.info(f"   Filling {expected_days - original_days} missing dates...")
            
            # Create full date dataframe
            full_df = pd.DataFrame({'ds': full_date_range})
            
            # Merge with original data
            df = full_df.merge(df, on='ds', how='left')
            
            # Fill missing values with interpolation for 'y'
            df['y'] = df['y'].interpolate(method='linear')
            
            # For remaining NaNs at start, use backfill
            df['y'] = df['y'].bfill()
            
            # Round to integers (admissions are whole numbers)
            df['y'] = df['y'].round().astype(int)
            
            # Fill other columns
            df['is_weekend'] = df['ds'].dt.dayofweek.isin([5, 6]).astype(int)
            df['month'] = df['ds'].dt.month
            
            # Fill disease_severity with forward fill then mean
            if 'disease_severity' in df.columns:
                df['disease_severity'] = df['disease_severity'].ffill().bfill()
                if df['disease_severity'].isna().any():
                    df['disease_severity'] = df['disease_severity'].fillna(df['disease_severity'].mean())
        
        return df
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame], aggregated_df: pd.DataFrame):
        """Save processed data to files"""
        logger.info("💾 Saving processed data...")
        
        # Save aggregated data
        aggregated_df.to_csv('data/processed/aggregated_admissions.csv', index=False)
        
        # Save forecast-ready data for each location
        for location, location_data in data.items():
            filename = f"data/processed/forecast_{location.replace(' ', '_').lower()}.csv"
            location_data.to_csv(filename, index=False)
        
        # Save combined forecast data
        if data:  # Check if data is not empty
            combined_forecast = pd.concat([
                df.assign(location=location) for location, df in data.items()
            ], ignore_index=True)
            
            combined_forecast.to_csv('data/processed/combined_forecast_data.csv', index=False)
        else:
            logger.warning("⚠️ No forecast data to save - creating empty file")
            # Create empty file with correct structure
            empty_df = pd.DataFrame(columns=['ds', 'y', 'location'])
            empty_df.to_csv('data/processed/combined_forecast_data.csv', index=False)
        
        logger.info("✅ All processed data saved")
    
    def generate_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """Generate data quality report"""
        logger.info("📋 Generating data quality report...")
        
        report = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().strftime('%Y-%m-%d'),
                'end': df['date'].max().strftime('%Y-%m-%d'),
                'days': (df['date'].max() - df['date'].min()).days
            },
            'locations': df['location'].nunique(),
            'missing_values': df.isnull().sum().to_dict(),
            'statistics': {
                'mean_daily_admissions': df['admission_count'].mean(),
                'median_daily_admissions': df['admission_count'].median(),
                'std_daily_admissions': df['admission_count'].std(),
                'min_admissions': df['admission_count'].min(),
                'max_admissions': df['admission_count'].max()
            },
            'location_summary': df.groupby('location')['admission_count'].agg(['count', 'mean', 'std']).to_dict()
        }
        
        # Save report
        import json
        os.makedirs('reports', exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert report to JSON-serializable format
        json_report = convert_numpy_types(report)
        
        with open('reports/data_quality_report.json', 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        logger.info("✅ Data quality report generated")
        
        return report

def main():
    """Main preprocessing function"""
    logger.info("🏥 Hospital Data Preprocessing Started")
    logger.info("=" * 40)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    admission_df, capacity_df = preprocessor.load_data()
    
    # Clean data
    cleaned_df = preprocessor.clean_data(admission_df)
    
    # Feature engineering
    featured_df = preprocessor.feature_engineering(cleaned_df)
    
    # Aggregate data
    aggregated_df = preprocessor.aggregate_data(featured_df)
    
    # Prepare forecast data
    forecast_data = preprocessor.prepare_forecast_data(aggregated_df)
    
    # Save processed data
    preprocessor.save_processed_data(forecast_data, aggregated_df)
    
    # Generate quality report
    quality_report = preprocessor.generate_data_quality_report(aggregated_df)
    
    # Print summary
    logger.info("\n📊 Processing Summary:")
    logger.info(f"- Total records processed: {quality_report['total_records']:,}")
    logger.info(f"- Date range: {quality_report['date_range']['start']} to {quality_report['date_range']['end']}")
    logger.info(f"- Locations: {quality_report['locations']}")
    logger.info(f"- Average daily admissions: {quality_report['statistics']['mean_daily_admissions']:.1f}")
    
    logger.info("\n🎉 Data preprocessing completed!")
    logger.info("\n📋 Next steps:")
    logger.info("1. jupyter notebook notebooks/02_model_development.ipynb")
    logger.info("2. python src/models/prophet_forecasting.py")

if __name__ == "__main__":
    main()