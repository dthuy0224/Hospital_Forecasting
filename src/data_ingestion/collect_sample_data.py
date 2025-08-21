#!/usr/bin/env python3
"""
Data Collection Script for Hospital Forecasting Project
Generates sample hospital admission and capacity data for demonstration
"""

import pandas as pd
import numpy as np
import sqlite3
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HospitalDataCollector:
    """Collect and generate hospital data for forecasting"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self.load_config(config_path)
        self.db_path = "data/hospital_forecasting.db"
        self.locations = self.config.get('locations', [
            "Ho Chi Minh City", "Ha Noi", "Da Nang", "Can Tho", "Hai Phong",
            "Nha Trang", "Hue", "Vung Tau", "Bien Hoa", "Thu Dau Mot"
        ])
        self.disease_types = [
            "COVID-19", "Flu", "Pneumonia", "Heart Disease", 
            "Diabetes", "Dengue Fever", "Others"
        ]
        self.age_groups = ["0-18", "19-30", "31-50", "51-65", "65+"]
        
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
            'locations': [
                "Ho Chi Minh City", "Ha Noi", "Da Nang", "Can Tho", "Hai Phong",
                "Nha Trang", "Hue", "Vung Tau", "Bien Hoa", "Thu Dau Mot"
            ],
            'data_sources': {
                'covid_data': {
                    'url': "https://disease.sh/v3/covid-19/historical",
                    'format': "json"
                }
            }
        }
    
    def create_database_connection(self) -> sqlite3.Connection:
        """Create database connection"""
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        logger.info(f"✅ Connected to database: {self.db_path}")
        return conn
    
    def generate_admission_data(self, start_date: str = "2023-01-01", 
                               end_date: str = "2024-12-31") -> pd.DataFrame:
        """Generate realistic hospital admission data"""
        logger.info("📊 Generating hospital admission data...")
        
        # Date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        data = []
        
        for location in self.locations:
            # Base admission rates per location (population-based)
            base_rates = {
                "Ho Chi Minh City": 150,
                "Ha Noi": 120,
                "Da Nang": 80,
                "Can Tho": 60,
                "Hai Phong": 70,
                "Nha Trang": 40,
                "Hue": 45,
                "Vung Tau": 35,
                "Bien Hoa": 50,
                "Thu Dau Mot": 30
            }
            
            base_rate = base_rates.get(location, 50)
            
            for date in dates:
                # Seasonal patterns
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                # Weekly patterns (weekends have fewer admissions)
                weekend_factor = 0.7 if date.weekday() >= 5 else 1.0
                
                # Holiday effect (simplified)
                holiday_factor = 0.6 if self.is_holiday(date) else 1.0
                
                # Disease-specific patterns
                for disease in self.disease_types:
                    disease_factor = self.get_disease_factor(disease, date)
                    
                    # Calculate base admissions for this disease
                    base_admissions = base_rate * disease_factor
                    
                    # Apply all factors
                    daily_admissions = int(base_admissions * seasonal_factor * 
                                         weekend_factor * holiday_factor)
                    
                    # Add realistic noise
                    noise = np.random.normal(0, daily_admissions * 0.2)
                    daily_admissions = max(0, int(daily_admissions + noise))
                    
                    # Distribute across age groups
                    for age_group in self.age_groups:
                        age_factor = self.get_age_factor(age_group, disease)
                        age_admissions = max(0, int(daily_admissions * age_factor))
                        
                        if age_admissions > 0:
                            data.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'location': location,
                                'admission_count': age_admissions,
                                'disease_type': disease,
                                'age_group': age_group,
                                'created_at': datetime.now().isoformat()
                            })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Generated {len(df):,} admission records")
        return df
    
    def get_disease_factor(self, disease: str, date: datetime) -> float:
        """Get disease-specific factors"""
        factors = {
            "COVID-19": 0.3,  # Lower now, but was higher in 2020-2022
            "Flu": 0.4,
            "Pneumonia": 0.25,
            "Heart Disease": 0.2,
            "Diabetes": 0.15,
            "Dengue Fever": 0.1,
            "Others": 0.5
        }
        
        # Add seasonal effects for specific diseases
        if disease == "Flu":
            # Flu season (winter months)
            if date.month in [12, 1, 2, 3]:
                return factors[disease] * 2.0
        elif disease == "Dengue Fever":
            # Dengue season (rainy season)
            if date.month in [6, 7, 8, 9, 10]:
                return factors[disease] * 3.0
        
        return factors.get(disease, 0.3)
    
    def get_age_factor(self, age_group: str, disease: str) -> float:
        """Get age-specific factors for diseases"""
        age_factors = {
            "0-18": 0.2,
            "19-30": 0.25,
            "31-50": 0.3,
            "51-65": 0.15,
            "65+": 0.1
        }
        
        # Disease-specific age adjustments
        if disease == "COVID-19":
            age_factors["65+"] = 0.25  # Higher risk for elderly
        elif disease == "Heart Disease":
            age_factors["51-65"] = 0.3
            age_factors["65+"] = 0.35
        elif disease == "Flu":
            age_factors["0-18"] = 0.3
            age_factors["65+"] = 0.2
        
        return age_factors.get(age_group, 0.2)
    
    def is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday (simplified)"""
        # Vietnamese holidays (simplified list)
        holidays = [
            (1, 1),   # New Year
            (4, 30),  # Reunification Day
            (5, 1),   # Labor Day
            (9, 2),   # National Day
        ]
        
        return (date.month, date.day) in holidays
    
    def generate_capacity_data(self, start_date: str = "2023-01-01", 
                              end_date: str = "2024-12-31") -> pd.DataFrame:
        """Generate hospital capacity data"""
        logger.info("🏥 Generating hospital capacity data...")
        
        # Date range
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        
        data = []
        
        # Base capacity per location
        base_capacities = {
            "Ho Chi Minh City": {"total": 5000, "icu": 500},
            "Ha Noi": {"total": 4000, "icu": 400},
            "Da Nang": {"total": 1500, "icu": 150},
            "Can Tho": {"total": 1200, "icu": 120},
            "Hai Phong": {"total": 1800, "icu": 180},
            "Nha Trang": {"total": 800, "icu": 80},
            "Hue": {"total": 900, "icu": 90},
            "Vung Tau": {"total": 600, "icu": 60},
            "Bien Hoa": {"total": 1000, "icu": 100},
            "Thu Dau Mot": {"total": 700, "icu": 70}
        }
        
        for location in self.locations:
            capacity = base_capacities.get(location, {"total": 500, "icu": 50})
            total_beds = capacity["total"]
            icu_beds = capacity["icu"]
            
            for date in dates:
                # Simulate some variation in available beds
                # (due to maintenance, seasonal adjustments, etc.)
                variation = np.random.normal(0, total_beds * 0.05)
                available_beds = max(0, int(total_beds + variation))
                
                data.append({
                    'location': location,
                    'total_beds': total_beds,
                    'icu_beds': icu_beds,
                    'available_beds': available_beds,
                    'date': date.strftime('%Y-%m-%d'),
                    'created_at': datetime.now().isoformat()
                })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Generated {len(df):,} capacity records")
        return df
    
    def fetch_covid_data(self) -> pd.DataFrame:
        """Fetch COVID-19 data from external API (optional)"""
        logger.info("🌐 Attempting to fetch COVID-19 data...")
        
        try:
            # Try to fetch Vietnam COVID data
            url = "https://disease.sh/v3/covid-19/historical/Vietnam?lastdays=365"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                timeline = data.get('timeline', {})
                cases = timeline.get('cases', {})
                
                covid_data = []
                for date_str, case_count in cases.items():
                    covid_data.append({
                        'date': date_str,
                        'location': 'Vietnam',
                        'admission_count': case_count,
                        'disease_type': 'COVID-19',
                        'age_group': 'Unknown',
                        'created_at': datetime.now().isoformat()
                    })
                
                df = pd.DataFrame(covid_data)
                logger.info(f"✅ Fetched {len(df)} COVID-19 records")
                return df
            else:
                logger.warning(f"⚠️ Failed to fetch COVID data: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch COVID data: {str(e)}")
            return pd.DataFrame()
    
    def save_to_database(self, admission_df: pd.DataFrame, capacity_df: pd.DataFrame):
        """Save data to SQLite database"""
        logger.info("💾 Saving data to database...")
        
        conn = self.create_database_connection()
        
        try:
            # Save admission data
            admission_df.to_sql('hospital_admissions', conn, if_exists='replace', index=False)
            logger.info(f"✅ Saved {len(admission_df):,} admission records")
            
            # Save capacity data
            capacity_df.to_sql('hospital_capacity', conn, if_exists='replace', index=False)
            logger.info(f"✅ Saved {len(capacity_df):,} capacity records")
            
        except Exception as e:
            logger.error(f"❌ Error saving to database: {str(e)}")
            raise
        finally:
            conn.close()
    
    def save_to_csv(self, admission_df: pd.DataFrame, capacity_df: pd.DataFrame):
        """Save data to CSV files"""
        logger.info("📄 Saving data to CSV files...")
        
        # Ensure directories exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Save admission data
        admission_csv_path = 'data/raw/hospital_admissions.csv'
        admission_df.to_csv(admission_csv_path, index=False)
        logger.info(f"✅ Saved admissions to: {admission_csv_path}")
        
        # Save capacity data
        capacity_csv_path = 'data/raw/hospital_capacity.csv'
        capacity_df.to_csv(capacity_csv_path, index=False)
        logger.info(f"✅ Saved capacity to: {capacity_csv_path}")
        
        # Save combined data for easy access
        combined_path = 'data/raw/combined_hospital_data.csv'
        combined_df = pd.concat([
            admission_df.assign(data_type='admissions'),
            capacity_df.assign(data_type='capacity')
        ], ignore_index=True)
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"✅ Saved combined data to: {combined_path}")
    
    def generate_data_summary(self, admission_df: pd.DataFrame, 
                            capacity_df: pd.DataFrame) -> Dict:
        """Generate summary statistics"""
        logger.info("📊 Generating data summary...")
        
        summary = {
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
                'avg_daily_admissions': admission_df['admission_count'].mean(),
                'max_daily_admissions': admission_df['admission_count'].max(),
                'total_admissions': admission_df['admission_count'].sum(),
                'avg_total_beds': capacity_df['total_beds'].mean(),
                'avg_icu_beds': capacity_df['icu_beds'].mean()
            }
        }
        
        # Save summary
        os.makedirs('reports', exist_ok=True)
        summary_path = 'reports/data_collection_summary.json'
        
        import json
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
        
        # Convert summary to JSON-serializable format
        json_summary = convert_numpy_types(summary)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Saved summary to: {summary_path}")
        return summary
    
    def collect_all_data(self, start_date: str = "2023-01-01", 
                        end_date: str = "2024-12-31",
                        include_covid_api: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect all hospital data"""
        logger.info("🏥 Hospital Data Collection Started")
        logger.info("=" * 50)
        
        # Generate synthetic data
        admission_df = self.generate_admission_data(start_date, end_date)
        capacity_df = self.generate_capacity_data(start_date, end_date)
        
        # Optionally fetch COVID data
        if include_covid_api:
            covid_df = self.fetch_covid_data()
            if not covid_df.empty:
                # Merge with synthetic data (simplified)
                admission_df = pd.concat([admission_df, covid_df], ignore_index=True)
                logger.info("✅ Integrated COVID-19 API data")
        
        # Save data
        self.save_to_database(admission_df, capacity_df)
        self.save_to_csv(admission_df, capacity_df)
        
        # Generate summary
        summary = self.generate_data_summary(admission_df, capacity_df)
        
        # Print summary
        logger.info("\n📊 Data Collection Summary:")
        logger.info(f"- Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        logger.info(f"- Locations: {summary['locations']['count']} ({', '.join(summary['locations']['list'][:3])}...)")
        logger.info(f"- Diseases: {summary['diseases']['count']} types")
        logger.info(f"- Total records: {summary['records']['total']:,}")
        logger.info(f"- Average daily admissions: {summary['statistics']['avg_daily_admissions']:.1f}")
        logger.info(f"- Average total beds: {summary['statistics']['avg_total_beds']:.0f}")
        
        logger.info("\n🎉 Data collection completed!")
        logger.info("\n📋 Next steps:")
        logger.info("1. python src/data_processing/preprocess_data.py")
        logger.info("2. python src/models/prophet_forecasting.py")
        logger.info("3. streamlit run src/visualization/streamlit_dashboard.py")
        
        return admission_df, capacity_df

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect hospital data for forecasting')
    parser.add_argument('--start-date', default='2023-01-01', 
                       help='Start date for data generation (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-12-31',
                       help='End date for data generation (YYYY-MM-DD)')
    parser.add_argument('--include-covid-api', action='store_true',
                       help='Include COVID-19 data from external API')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = HospitalDataCollector(args.config)
    
    # Collect data
    try:
        admission_df, capacity_df = collector.collect_all_data(
            start_date=args.start_date,
            end_date=args.end_date,
            include_covid_api=args.include_covid_api
        )
        
        print(f"\n🎉 SUCCESS! Generated {len(admission_df):,} admission records and {len(capacity_df):,} capacity records")
        print(f"📁 Data saved to:")
        print(f"   - Database: {collector.db_path}")
        print(f"   - CSV files: data/raw/")
        print(f"   - Summary: reports/data_collection_summary.json")
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 