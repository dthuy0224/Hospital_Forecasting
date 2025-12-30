#!/usr/bin/env python3
"""
Kaggle Data Collector for Hospital Forecasting Project
Downloads and transforms real hospital admission data from Kaggle
"""

import pandas as pd
import numpy as np
import sqlite3
import yaml
import logging
import os
import json
import zipfile
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KaggleDataCollector:
    """Collect real hospital data from Kaggle datasets"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self.load_config(config_path)
        self.db_path = "data/hospital_forecasting.db"
        self.raw_data_path = "data/raw"
        self.kaggle_cache_path = "data/kaggle_cache"
        
        # Disease type mapping from medical conditions
        self.disease_mapping = {
            # Heart conditions
            'CAD': 'Heart Disease',
            'STEMI': 'Heart Disease',
            'ACS': 'Heart Disease',
            'HTN': 'Heart Disease',
            'CMP': 'Heart Disease',
            'CHB': 'Heart Disease',
            'AF': 'Heart Disease',
            'VT': 'Heart Disease',
            'PSVT': 'Heart Disease',
            'WPWS': 'Heart Disease',
            'CHF': 'Heart Disease',
            # Other conditions
            'DM': 'Diabetes',
            'CKD': 'Kidney Disease',
            'UTI': 'Infection',
            'INFECTIVE ENDOCARDITIS': 'Infection',
            'DVT': 'Blood Clot',
            'CVA': 'Stroke',
            'NEURO': 'Neurological',
        }
        
        # Age group bins
        self.age_bins = [0, 18, 30, 50, 65, 120]
        self.age_labels = ["0-18", "19-30", "31-50", "51-65", "65+"]
        
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
            'kaggle': {
                'datasets': {
                    'hospital_admissions': {
                        'name': 'ashishsahani/hospital-admissions-data',
                        'file': 'HDHI Admission data.csv'
                    }
                }
            }
        }
    
    def setup_kaggle_credentials(self, api_key: str, username: str = None):
        """Setup Kaggle API credentials"""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_json = kaggle_dir / 'kaggle.json'
        
        # Create .kaggle directory
        kaggle_dir.mkdir(exist_ok=True)
        
        # If full key provided in format "username:key" or just key
        if ':' in api_key:
            username, key = api_key.split(':')
        else:
            # For API token format (KGAT_xxx), we need username separately
            key = api_key
            if username is None:
                username = os.environ.get('KAGGLE_USERNAME', 'default_user')
        
        credentials = {
            'username': username,
            'key': key
        }
        
        # Write credentials file
        with open(kaggle_json, 'w') as f:
            json.dump(credentials, f)
        
        # Set permissions (Unix only)
        try:
            os.chmod(kaggle_json, 0o600)
        except:
            pass
        
        logger.info(f"✅ Kaggle credentials saved to {kaggle_json}")
        
    def download_kaggle_dataset(self, dataset_name: str = "ashishsahani/hospital-admissions-data") -> str:
        """Download dataset from Kaggle"""
        logger.info(f"📥 Downloading Kaggle dataset: {dataset_name}")
        
        # Create cache directory
        os.makedirs(self.kaggle_cache_path, exist_ok=True)
        
        try:
            # Try to import kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            api.dataset_download_files(
                dataset_name, 
                path=self.kaggle_cache_path, 
                unzip=True
            )
            
            logger.info(f"✅ Dataset downloaded to {self.kaggle_cache_path}")
            return self.kaggle_cache_path
            
        except ImportError:
            logger.error("❌ Kaggle package not installed. Run: pip install kaggle")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to download dataset: {str(e)}")
            raise
    
    def load_kaggle_data(self, file_path: str = None) -> pd.DataFrame:
        """Load downloaded Kaggle data"""
        if file_path is None:
            # Look for the CSV file in cache
            file_path = os.path.join(self.kaggle_cache_path, 'HDHI Admission data.csv')
        
        if not os.path.exists(file_path):
            logger.error(f"❌ Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"📖 Loading data from {file_path}")
        
        # Read CSV with proper encoding
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        logger.info(f"✅ Loaded {len(df):,} records with {len(df.columns)} columns")
        
        return df
    
    def transform_to_admission_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform Kaggle data to match our admission schema"""
        logger.info("🔄 Transforming data to admission schema...")
        
        transformed_data = []
        
        # Parse date column - handle multiple date formats
        date_col = None
        for col in ['D.O.A', 'DOA', 'Date of Admission', 'date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            # Try to find any column with 'date' in name
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'd.o.a' in c.lower()]
            if date_cols:
                date_col = date_cols[0]
            else:
                logger.warning("⚠️ No date column found, using index as date")
                df['parsed_date'] = pd.date_range(start='2017-04-01', periods=len(df), freq='D')
        
        if date_col:
            # Try to parse dates
            df['parsed_date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
        
        # Get age column
        age_col = None
        for col in ['AGE', 'Age', 'age']:
            if col in df.columns:
                age_col = col
                break
        
        # Ensure age is numeric
        if age_col:
            df['age_numeric'] = pd.to_numeric(df[age_col], errors='coerce').fillna(35)
        else:
            df['age_numeric'] = 35  # Default age
        
        # Bin ages into groups
        df['age_group'] = pd.cut(
            df['age_numeric'], 
            bins=self.age_bins, 
            labels=self.age_labels, 
            right=False
        )
        
        # Determine disease type from medical conditions
        df['disease_type'] = df.apply(lambda row: self._infer_disease_type(row), axis=1)
        
        # Get location - use hospital regions or create virtual regions
        df['location'] = self._assign_locations(df)
        
        # Group by date, location, disease_type, age_group and count
        df_valid = df.dropna(subset=['parsed_date'])
        
        if len(df_valid) == 0:
            logger.error("❌ No valid dates found in data")
            raise ValueError("No valid dates found in data")
        
        grouped = df_valid.groupby([
            df_valid['parsed_date'].dt.date,
            'location',
            'disease_type',
            'age_group'
        ]).size().reset_index(name='admission_count')
        
        grouped.columns = ['date', 'location', 'disease_type', 'age_group', 'admission_count']
        grouped['date'] = pd.to_datetime(grouped['date']).dt.strftime('%Y-%m-%d')
        grouped['created_at'] = datetime.now().isoformat()
        
        logger.info(f"✅ Transformed to {len(grouped):,} aggregated records")
        return grouped
    
    def _infer_disease_type(self, row) -> str:
        """Infer disease type from row data"""
        # Check for specific conditions in the row
        for col in row.index:
            col_upper = str(col).upper()
            value = row[col]
            
            # Check if column indicates a condition and value is positive
            if pd.notna(value):
                try:
                    # Check for binary indicators (1, Yes, Y, True)
                    if str(value).strip().upper() in ['1', 'YES', 'Y', 'TRUE']:
                        # Check column name against disease mapping
                        for key, disease in self.disease_mapping.items():
                            if key in col_upper:
                                return disease
                except:
                    pass
        
        # Check for TYPE OF ADMISSION or FINAL DIAGNOSIS columns
        for col in row.index:
            if 'DIAGNOSIS' in str(col).upper() or 'TYPE' in str(col).upper():
                value = str(row[col]).upper() if pd.notna(row[col]) else ''
                for key, disease in self.disease_mapping.items():
                    if key in value:
                        return disease
        
        # Default to Heart Disease (since this is a heart institute dataset)
        return "Heart Disease"
    
    def _assign_locations(self, df: pd.DataFrame) -> pd.Series:
        """Assign virtual locations based on data distribution"""
        # Since HDHI is a single hospital, we'll create virtual regions
        # based on RURAL column or other demographics
        
        locations = [
            "Ludhiana Central",  # Main hospital location
            "Ludhiana South",
            "Ludhiana North", 
            "Punjab Rural",
            "Punjab Urban"
        ]
        
        # Check if RURAL column exists
        rural_col = None
        for col in df.columns:
            if 'RURAL' in str(col).upper():
                rural_col = col
                break
        
        if rural_col and rural_col in df.columns:
            # Assign based on rural/urban
            def assign_location(row):
                if pd.notna(row.get(rural_col)):
                    if str(row[rural_col]).upper() in ['1', 'YES', 'Y', 'RURAL']:
                        return np.random.choice(["Punjab Rural", "Ludhiana North"])
                    else:
                        return np.random.choice(["Ludhiana Central", "Ludhiana South", "Punjab Urban"])
                return np.random.choice(locations)
            
            return df.apply(assign_location, axis=1)
        else:
            # Random assignment with weighted distribution
            np.random.seed(42)  # For reproducibility
            weights = [0.4, 0.2, 0.15, 0.15, 0.1]
            return pd.Series(np.random.choice(locations, size=len(df), p=weights))
    
    def generate_capacity_data(self, admission_df: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic capacity data based on admissions"""
        logger.info("🏥 Generating capacity data based on admissions...")
        
        locations = admission_df['location'].unique()
        dates = pd.to_datetime(admission_df['date']).unique()
        
        # Base capacity per location
        base_capacities = {
            "Ludhiana Central": {"total": 500, "icu": 50},
            "Ludhiana South": {"total": 300, "icu": 30},
            "Ludhiana North": {"total": 250, "icu": 25},
            "Punjab Rural": {"total": 200, "icu": 20},
            "Punjab Urban": {"total": 350, "icu": 35},
        }
        
        data = []
        for location in locations:
            capacity = base_capacities.get(location, {"total": 300, "icu": 30})
            
            for date in dates:
                # Add some variation
                variation = np.random.normal(0, capacity["total"] * 0.05)
                available = max(0, int(capacity["total"] + variation))
                
                data.append({
                    'location': location,
                    'total_beds': capacity["total"],
                    'icu_beds': capacity["icu"],
                    'available_beds': available,
                    'date': pd.to_datetime(date).strftime('%Y-%m-%d'),
                    'created_at': datetime.now().isoformat()
                })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Generated {len(df):,} capacity records")
        return df
    
    def save_to_database(self, admission_df: pd.DataFrame, capacity_df: pd.DataFrame):
        """Save data to SQLite database"""
        logger.info("💾 Saving data to database...")
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        
        try:
            admission_df.to_sql('hospital_admissions', conn, if_exists='replace', index=False)
            logger.info(f"✅ Saved {len(admission_df):,} admission records")
            
            capacity_df.to_sql('hospital_capacity', conn, if_exists='replace', index=False)
            logger.info(f"✅ Saved {len(capacity_df):,} capacity records")
            
        finally:
            conn.close()
    
    def save_to_csv(self, admission_df: pd.DataFrame, capacity_df: pd.DataFrame):
        """Save data to CSV files"""
        logger.info("📄 Saving data to CSV files...")
        
        os.makedirs(self.raw_data_path, exist_ok=True)
        
        admission_path = os.path.join(self.raw_data_path, 'hospital_admissions.csv')
        admission_df.to_csv(admission_path, index=False)
        logger.info(f"✅ Saved admissions to: {admission_path}")
        
        capacity_path = os.path.join(self.raw_data_path, 'hospital_capacity.csv')
        capacity_df.to_csv(capacity_path, index=False)
        logger.info(f"✅ Saved capacity to: {capacity_path}")
    
    def generate_data_summary(self, admission_df: pd.DataFrame, 
                             capacity_df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        logger.info("📊 Generating data summary...")
        
        summary = {
            'data_source': 'Kaggle HDHI Hospital Admissions',
            'generated_at': datetime.now().isoformat(),
            'date_range': {
                'start': admission_df['date'].min(),
                'end': admission_df['date'].max(),
                'days': len(admission_df['date'].unique())
            },
            'locations': {
                'count': admission_df['location'].nunique(),
                'list': sorted(admission_df['location'].unique().tolist())
            },
            'diseases': {
                'count': admission_df['disease_type'].nunique(),
                'list': sorted(admission_df['disease_type'].unique().tolist())
            },
            'age_groups': {
                'count': admission_df['age_group'].nunique(),
                'list': sorted(admission_df['age_group'].unique().tolist())
            },
            'records': {
                'admissions': len(admission_df),
                'capacity': len(capacity_df),
                'total': len(admission_df) + len(capacity_df)
            },
            'statistics': {
                'avg_daily_admissions': float(admission_df['admission_count'].mean()),
                'max_daily_admissions': int(admission_df['admission_count'].max()),
                'total_admissions': int(admission_df['admission_count'].sum()),
                'avg_total_beds': float(capacity_df['total_beds'].mean()),
                'avg_icu_beds': float(capacity_df['icu_beds'].mean())
            }
        }
        
        # Save summary
        os.makedirs('reports', exist_ok=True)
        summary_path = 'reports/data_collection_summary.json'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved summary to: {summary_path}")
        return summary
    
    def collect_kaggle_data(self, 
                           dataset_name: str = "ashishsahani/hospital-admissions-data",
                           force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Main method to collect and process Kaggle data"""
        logger.info("🏥 Kaggle Data Collection Started")
        logger.info("=" * 50)
        
        # Check if data already exists in cache
        cached_file = os.path.join(self.kaggle_cache_path, 'HDHI Admission data.csv')
        
        if not os.path.exists(cached_file) or force_download:
            # Download from Kaggle
            self.download_kaggle_dataset(dataset_name)
        else:
            logger.info(f"📦 Using cached data from {cached_file}")
        
        # Load raw data
        raw_df = self.load_kaggle_data(cached_file)
        
        # Transform to our schema
        admission_df = self.transform_to_admission_schema(raw_df)
        
        # Generate capacity data
        capacity_df = self.generate_capacity_data(admission_df)
        
        # Save data
        self.save_to_database(admission_df, capacity_df)
        self.save_to_csv(admission_df, capacity_df)
        
        # Generate summary
        summary = self.generate_data_summary(admission_df, capacity_df)
        
        # Print summary
        logger.info("\n📊 Data Collection Summary:")
        logger.info(f"- Data source: Kaggle HDHI Hospital Admissions")
        logger.info(f"- Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        logger.info(f"- Locations: {summary['locations']['count']} ({', '.join(summary['locations']['list'][:3])}...)")
        logger.info(f"- Diseases: {summary['diseases']['count']} types")
        logger.info(f"- Total records: {summary['records']['total']:,}")
        logger.info(f"- Total admissions: {summary['statistics']['total_admissions']:,}")
        
        logger.info("\n🎉 Kaggle data collection completed!")
        logger.info("\n📋 Next steps:")
        logger.info("1. python src/data_processing/preprocess_data.py")
        logger.info("2. python src/models/prophet_forecasting.py")
        logger.info("3. streamlit run src/visualization/streamlit_dashboard.py")
        
        return admission_df, capacity_df


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect hospital data from Kaggle')
    parser.add_argument('--dataset', default='ashishsahani/hospital-admissions-data',
                       help='Kaggle dataset name')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download even if cached')
    parser.add_argument('--kaggle-key', type=str,
                       help='Kaggle API key')
    parser.add_argument('--kaggle-username', type=str,
                       help='Kaggle username')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = KaggleDataCollector(args.config)
    
    # Setup credentials if provided
    if args.kaggle_key:
        collector.setup_kaggle_credentials(args.kaggle_key, args.kaggle_username)
    
    # Collect data
    try:
        admission_df, capacity_df = collector.collect_kaggle_data(
            dataset_name=args.dataset,
            force_download=args.force_download
        )
        
        print(f"\n[SUCCESS] Collected {len(admission_df):,} admission records")
        print(f"[DATA] saved to:")
        print(f"   - Database: {collector.db_path}")
        print(f"   - CSV files: {collector.raw_data_path}/")
        print(f"   - Summary: reports/data_collection_summary.json")
        
    except Exception as e:
        logger.error(f"[FAILED] Data collection failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
