#!/usr/bin/env python3
"""
Advanced Feature Engineering Script for Hospital Forecasting Project
Implements sophisticated features like holidays, weather patterns, demographic factors
"""

import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import requests
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for time series forecasting"""
    
    def __init__(self, data_path: str = "data/processed/combined_forecast_data.csv"):
        self.data_path = data_path
        self.enhanced_data = {}
        
        # Vietnamese holidays and special events
        self.vietnamese_holidays = {
            'tet_lunar_new_year': ['2023-01-22', '2023-01-23', '2023-01-24', '2023-01-25', '2023-01-26',
                                  '2024-02-10', '2024-02-11', '2024-02-12', '2024-02-13', '2024-02-14'],
            'hung_kings_day': ['2023-04-29', '2024-04-18'],
            'liberation_day': ['2023-04-30', '2024-04-30'],
            'labor_day': ['2023-05-01', '2024-05-01'],
            'national_day': ['2023-09-02', '2024-09-02'],
            'mid_autumn_festival': ['2023-09-29', '2024-09-17']
        }
        
        # Population data for major cities (thousands)
        self.population_data = {
            'Ho Chi Minh City': 9000,
            'Ha Noi': 8000,
            'Hai Phong': 2000,
            'Da Nang': 1200,
            'Can Tho': 1200
        }
        
        # Climate zones for weather patterns
        self.climate_zones = {
            'Ho Chi Minh City': 'tropical',
            'Ha Noi': 'subtropical',
            'Hai Phong': 'subtropical',
            'Da Nang': 'tropical_monsoon',
            'Can Tho': 'tropical'
        }
        
    def load_data(self) -> pd.DataFrame:
        """Load and prepare base data"""
        logger.info("📊 Loading base data for feature engineering...")
        
        try:
            df = pd.read_csv(self.data_path)
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.sort_values(['location', 'ds'])
            
            logger.info(f"✅ Loaded {len(df)} records for feature engineering")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Vietnamese holiday and special event features"""
        logger.info("🎉 Adding holiday features...")
        
        try:
            # Convert date to string for comparison
            df['date_str'] = df['ds'].dt.strftime('%Y-%m-%d')
            
            # Add holiday indicators
            df['is_tet_period'] = df['date_str'].isin(self.vietnamese_holidays['tet_lunar_new_year']).astype(int)
            df['is_hung_kings_day'] = df['date_str'].isin(self.vietnamese_holidays['hung_kings_day']).astype(int)
            df['is_liberation_day'] = df['date_str'].isin(self.vietnamese_holidays['liberation_day']).astype(int)
            df['is_labor_day'] = df['date_str'].isin(self.vietnamese_holidays['labor_day']).astype(int)
            df['is_national_day'] = df['date_str'].isin(self.vietnamese_holidays['national_day']).astype(int)
            df['is_mid_autumn'] = df['date_str'].isin(self.vietnamese_holidays['mid_autumn_festival']).astype(int)
            
            # Combine all holidays
            df['is_major_holiday'] = (df['is_tet_period'] | df['is_hung_kings_day'] | 
                                     df['is_liberation_day'] | df['is_labor_day'] | 
                                     df['is_national_day'] | df['is_mid_autumn']).astype(int)
            
            # Add pre/post holiday effects (3 days before and after)
            df['pre_holiday'] = 0
            df['post_holiday'] = 0
            
            for i in range(len(df)):
                current_date = df.iloc[i]['ds']
                
                # Check 3 days before and after for holiday effects
                for days_offset in range(-3, 4):
                    check_date = current_date + timedelta(days=days_offset)
                    check_date_str = check_date.strftime('%Y-%m-%d')
                    
                    # Check if it's a holiday
                    is_holiday = any(check_date_str in holidays for holidays in self.vietnamese_holidays.values())
                    
                    if is_holiday:
                        if days_offset < 0:  # Before holiday
                            df.iloc[i, df.columns.get_loc('pre_holiday')] = 1
                        elif days_offset > 0:  # After holiday
                            df.iloc[i, df.columns.get_loc('post_holiday')] = 1
            
            # Add Tet season effect (broader period)
            df['is_tet_season'] = 0
            for tet_date_str in self.vietnamese_holidays['tet_lunar_new_year']:
                tet_date = pd.to_datetime(tet_date_str)
                # 2 weeks before to 1 week after Tet
                tet_season_mask = ((df['ds'] >= tet_date - timedelta(days=14)) & 
                                  (df['ds'] <= tet_date + timedelta(days=7)))
                df.loc[tet_season_mask, 'is_tet_season'] = 1
            
            # Remove temporary column
            df = df.drop('date_str', axis=1)
            
            logger.info("✅ Holiday features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding holiday features: {str(e)}")
            return df
    
    def add_weather_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add weather and seasonal pattern features"""
        logger.info("🌤️ Adding weather and seasonal features...")
        
        try:
            # Basic seasonal features
            df['month'] = df['ds'].dt.month
            df['quarter'] = df['ds'].dt.quarter
            df['day_of_year'] = df['ds'].dt.dayofyear
            df['week_of_year'] = df['ds'].dt.isocalendar().week
            
            # Vietnamese seasonal patterns
            df['is_dry_season'] = ((df['month'] >= 11) | (df['month'] <= 4)).astype(int)  # Nov-Apr
            df['is_rainy_season'] = ((df['month'] >= 5) & (df['month'] <= 10)).astype(int)  # May-Oct
            
            # Monsoon patterns
            df['is_southwest_monsoon'] = ((df['month'] >= 5) & (df['month'] <= 9)).astype(int)  # May-Sep
            df['is_northeast_monsoon'] = ((df['month'] >= 10) | (df['month'] <= 3)).astype(int)  # Oct-Mar
            
            # Temperature seasonality (sine/cosine for cyclical patterns)
            df['temp_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
            df['temp_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
            
            # Humidity patterns (higher during rainy season)
            df['humidity_factor'] = np.where(df['is_rainy_season'] == 1, 1.3, 0.7)
            
            # Add climate zone-specific features
            for location in df['location'].unique():
                location_mask = df['location'] == location
                climate_zone = self.climate_zones.get(location, 'tropical')
                
                if climate_zone == 'tropical':
                    # More consistent temperature, high humidity
                    df.loc[location_mask, 'climate_temp_variation'] = 0.8
                    df.loc[location_mask, 'climate_humidity_base'] = 1.2
                elif climate_zone == 'subtropical':
                    # More temperature variation, seasonal humidity
                    df.loc[location_mask, 'climate_temp_variation'] = 1.5
                    df.loc[location_mask, 'climate_humidity_base'] = 1.0
                else:  # tropical_monsoon
                    # Strong seasonal patterns
                    df.loc[location_mask, 'climate_temp_variation'] = 1.2
                    df.loc[location_mask, 'climate_humidity_base'] = 1.4
            
            # Extreme weather indicators
            df['extreme_heat_risk'] = ((df['month'].isin([4, 5, 6])) & 
                                      (df['location'].isin(['Ho Chi Minh City', 'Da Nang']))).astype(int)
            
            df['flood_risk_season'] = ((df['month'].isin([7, 8, 9, 10])) & 
                                      (df['location'].isin(['Ho Chi Minh City', 'Can Tho']))).astype(int)
            
            logger.info("✅ Weather and seasonal features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding weather features: {str(e)}")
            return df
    
    def add_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add demographic and socioeconomic features"""
        logger.info("👥 Adding demographic features...")
        
        try:
            # Add population data
            df['population'] = df['location'].map(self.population_data).fillna(1000)  # Default 1M
            
            # Population density categories
            df['population_category'] = pd.cut(df['population'], 
                                             bins=[0, 1500, 3000, 10000], 
                                             labels=['small', 'medium', 'large'])
            
            # Urban vs rural characteristics (based on population)
            df['is_major_city'] = (df['population'] >= 3000).astype(int)
            df['is_coastal_city'] = df['location'].isin(['Ho Chi Minh City', 'Hai Phong', 'Da Nang']).astype(int)
            df['is_delta_region'] = df['location'].isin(['Ho Chi Minh City', 'Can Tho']).astype(int)
            
            # Economic activity indicators
            df['is_economic_hub'] = df['location'].isin(['Ho Chi Minh City', 'Ha Noi']).astype(int)
            df['is_industrial_city'] = df['location'].isin(['Hai Phong', 'Da Nang']).astype(int)
            df['is_tourism_city'] = df['location'].isin(['Da Nang', 'Ho Chi Minh City']).astype(int)
            
            # Healthcare infrastructure proxy (based on city size and importance)
            healthcare_capacity = {
                'Ho Chi Minh City': 1.5,  # Highest capacity
                'Ha Noi': 1.4,
                'Da Nang': 1.1,
                'Hai Phong': 1.0,
                'Can Tho': 0.9
            }
            df['healthcare_capacity_factor'] = df['location'].map(healthcare_capacity).fillna(1.0)
            
            # Age structure proxies (major cities have different age distributions)
            df['elderly_population_factor'] = np.where(df['is_major_city'] == 1, 1.2, 1.0)
            df['working_age_factor'] = np.where(df['is_economic_hub'] == 1, 1.3, 1.0)
            
            logger.info("✅ Demographic features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding demographic features: {str(e)}")
            return df
    
    def add_temporal_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add complex temporal interaction features"""
        logger.info("⏰ Adding temporal interaction features...")
        
        try:
            # Day of week interactions with seasons
            df['weekend_dry_season'] = (df['is_weekend'] * df['is_dry_season'])
            df['weekend_rainy_season'] = (df['is_weekend'] * df['is_rainy_season'])
            
            # Holiday interactions with weather
            df['holiday_dry_season'] = (df['is_major_holiday'] * df['is_dry_season'])
            df['holiday_rainy_season'] = (df['is_major_holiday'] * df['is_rainy_season'])
            
            # Population interactions with seasons
            df['large_city_rainy'] = (df['is_major_city'] * df['is_rainy_season'])
            df['coastal_monsoon'] = (df['is_coastal_city'] * df['is_southwest_monsoon'])
            
            # Economic activity interactions
            df['economic_hub_holiday'] = (df['is_economic_hub'] * df['is_major_holiday'])
            df['tourism_season'] = (df['is_tourism_city'] * df['is_dry_season'])  # Dry season = tourism season
            
            # Healthcare capacity interactions
            df['capacity_holiday_stress'] = (df['healthcare_capacity_factor'] * df['is_major_holiday'])
            df['capacity_weather_stress'] = (df['healthcare_capacity_factor'] * df['extreme_heat_risk'])
            
            # Cyclical interactions
            df['month_population'] = df['month'] * df['population'] / 1000  # Normalized
            df['quarter_economic'] = df['quarter'] * df['is_economic_hub']
            
            logger.info("✅ Temporal interaction features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding temporal interaction features: {str(e)}")
            return df
    
    def add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features and rolling statistics"""
        logger.info("📈 Adding lagged features and rolling statistics...")
        
        try:
            # Sort by location and date
            df = df.sort_values(['location', 'ds'])
            
            # Add lagged features for each location
            for location in df['location'].unique():
                location_mask = df['location'] == location
                location_data = df[location_mask].copy()
                
                # Lagged values (1, 7, 14, 30 days)
                for lag in [1, 7, 14, 30]:
                    df.loc[location_mask, f'y_lag_{lag}'] = location_data['y'].shift(lag)
                
                # Rolling statistics (7, 14, 30 days)
                for window in [7, 14, 30]:
                    df.loc[location_mask, f'y_rolling_mean_{window}'] = location_data['y'].rolling(window).mean()
                    df.loc[location_mask, f'y_rolling_std_{window}'] = location_data['y'].rolling(window).std()
                    df.loc[location_mask, f'y_rolling_min_{window}'] = location_data['y'].rolling(window).min()
                    df.loc[location_mask, f'y_rolling_max_{window}'] = location_data['y'].rolling(window).max()
                
                # Exponential moving averages
                for alpha in [0.1, 0.3, 0.5]:
                    df.loc[location_mask, f'y_ema_{alpha}'] = location_data['y'].ewm(alpha=alpha).mean()
                
                # Trend features
                df.loc[location_mask, 'y_trend_7d'] = (location_data['y'] - location_data['y'].shift(7))
                df.loc[location_mask, 'y_trend_30d'] = (location_data['y'] - location_data['y'].shift(30))
                
                # Volatility features
                df.loc[location_mask, 'y_volatility_7d'] = location_data['y'].rolling(7).std()
                df.loc[location_mask, 'y_volatility_30d'] = location_data['y'].rolling(30).std()
            
            logger.info("✅ Lagged features and rolling statistics added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding lagged features: {str(e)}")
            return df
    
    def add_fourier_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Fourier features for complex seasonality"""
        logger.info("🌊 Adding Fourier features for seasonality...")
        
        try:
            # Annual seasonality (multiple harmonics)
            for k in range(1, 6):  # 5 harmonics for annual pattern
                df[f'annual_sin_{k}'] = np.sin(2 * np.pi * k * df['day_of_year'] / 365.25)
                df[f'annual_cos_{k}'] = np.cos(2 * np.pi * k * df['day_of_year'] / 365.25)
            
            # Weekly seasonality (multiple harmonics)
            df['day_of_week'] = df['ds'].dt.dayofweek
            for k in range(1, 4):  # 3 harmonics for weekly pattern
                df[f'weekly_sin_{k}'] = np.sin(2 * np.pi * k * df['day_of_week'] / 7)
                df[f'weekly_cos_{k}'] = np.cos(2 * np.pi * k * df['day_of_week'] / 7)
            
            # Monthly seasonality
            for k in range(1, 3):  # 2 harmonics for monthly pattern
                df[f'monthly_sin_{k}'] = np.sin(2 * np.pi * k * df['ds'].dt.day / 30.44)
                df[f'monthly_cos_{k}'] = np.cos(2 * np.pi * k * df['ds'].dt.day / 30.44)
            
            # Quarterly seasonality
            quarter_day = ((df['ds'].dt.month - 1) % 3) * 30 + df['ds'].dt.day
            df['quarterly_sin'] = np.sin(2 * np.pi * quarter_day / 91.25)
            df['quarterly_cos'] = np.cos(2 * np.pi * quarter_day / 91.25)
            
            logger.info("✅ Fourier features added successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error adding Fourier features: {str(e)}")
            return df
    
    def engineer_all_features(self) -> pd.DataFrame:
        """Run complete advanced feature engineering pipeline"""
        logger.info("🚀 Starting advanced feature engineering...")
        
        try:
            # Load base data
            df = self.load_data()
            
            # Add all feature categories
            df = self.add_holiday_features(df)
            df = self.add_weather_seasonal_features(df)
            df = self.add_demographic_features(df)
            df = self.add_temporal_interaction_features(df)
            df = self.add_lagged_features(df)
            df = self.add_fourier_features(df)
            
            # Clean up any missing values created by lagged features
            # Forward fill for the first few rows
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"✅ Advanced feature engineering completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Advanced feature engineering failed: {str(e)}")
            raise
    
    def save_enhanced_data(self, df: pd.DataFrame):
        """Save enhanced dataset with advanced features"""
        logger.info("💾 Saving enhanced dataset...")
        
        try:
            # Create directories
            os.makedirs('data/processed', exist_ok=True)
            os.makedirs('reports', exist_ok=True)
            
            # Save enhanced dataset
            output_path = 'data/processed/enhanced_forecast_data.csv'
            df.to_csv(output_path, index=False)
            
            # Generate feature report
            feature_report = {
                'enhancement_date': datetime.now().isoformat(),
                'total_records': len(df),
                'total_features': len(df.columns),
                'locations': df['location'].unique().tolist(),
                'date_range': {
                    'start': str(df['ds'].min()),
                    'end': str(df['ds'].max()),
                    'days': len(df['ds'].unique())
                },
                'feature_categories': {
                    'holiday_features': [col for col in df.columns if 'holiday' in col or 'tet' in col],
                    'weather_features': [col for col in df.columns if any(x in col for x in ['season', 'monsoon', 'temp', 'humidity', 'climate'])],
                    'demographic_features': [col for col in df.columns if any(x in col for x in ['population', 'city', 'economic', 'healthcare'])],
                    'temporal_features': [col for col in df.columns if any(x in col for x in ['month', 'quarter', 'week', 'day'])],
                    'lagged_features': [col for col in df.columns if 'lag' in col or 'rolling' in col or 'ema' in col],
                    'fourier_features': [col for col in df.columns if 'sin' in col or 'cos' in col]
                },
                'missing_values': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.astype(str).to_dict()
            }
            
            # Save feature report
            with open('reports/feature_engineering_report.json', 'w', encoding='utf-8') as f:
                json.dump(feature_report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Enhanced dataset saved to {output_path}")
            logger.info(f"📊 Feature report saved to reports/feature_engineering_report.json")
            
        except Exception as e:
            logger.error(f"❌ Failed to save enhanced dataset: {str(e)}")
    
    def generate_feature_importance_analysis(self, df: pd.DataFrame):
        """Generate feature importance analysis using correlation and mutual information"""
        logger.info("📊 Generating feature importance analysis...")
        
        try:
            from sklearn.feature_selection import mutual_info_regression
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare data for analysis
            analysis_df = df.copy()
            
            # Encode categorical variables
            le = LabelEncoder()
            categorical_cols = analysis_df.select_dtypes(include=['object', 'category']).columns
            
            for col in categorical_cols:
                if col != 'ds':  # Skip datetime column
                    analysis_df[col] = le.fit_transform(analysis_df[col].astype(str))
            
            # Select numeric columns for analysis (exclude target and datetime)
            feature_cols = [col for col in analysis_df.columns if col not in ['y', 'ds']]
            X = analysis_df[feature_cols]
            y = analysis_df['y']
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Calculate correlations with target
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y)
            mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
            
            # Create importance analysis
            importance_analysis = {
                'analysis_date': datetime.now().isoformat(),
                'top_correlated_features': correlations.head(20).to_dict(),
                'top_mutual_info_features': mi_scores.head(20).to_dict(),
                'feature_statistics': {
                    'total_features': len(feature_cols),
                    'highly_correlated': len(correlations[correlations > 0.3]),
                    'moderately_correlated': len(correlations[(correlations > 0.1) & (correlations <= 0.3)]),
                    'low_correlated': len(correlations[correlations <= 0.1])
                },
                'recommendations': []
            }
            
            # Add recommendations based on analysis
            if len(correlations[correlations > 0.5]) > 0:
                importance_analysis['recommendations'].append(
                    f"Found {len(correlations[correlations > 0.5])} highly correlated features (>0.5) - consider for model training"
                )
            
            if len(correlations[correlations < 0.05]) > 10:
                importance_analysis['recommendations'].append(
                    f"Found {len(correlations[correlations < 0.05])} very low correlated features (<0.05) - consider removing"
                )
            
            # Save analysis
            with open('reports/feature_importance_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(importance_analysis, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("✅ Feature importance analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Feature importance analysis failed: {str(e)}")

def main():
    """Main function to run advanced feature engineering"""
    logger.info("🏥 Hospital Forecasting Advanced Feature Engineering Started")
    logger.info("=" * 60)
    
    try:
        # Initialize feature engineer
        engineer = AdvancedFeatureEngineer()
        
        # Run complete feature engineering
        enhanced_df = engineer.engineer_all_features()
        
        # Save enhanced dataset
        engineer.save_enhanced_data(enhanced_df)
        
        # Generate feature importance analysis
        engineer.generate_feature_importance_analysis(enhanced_df)
        
        # Print summary
        logger.info("📊 FEATURE ENGINEERING SUMMARY:")
        logger.info(f"Total records: {len(enhanced_df):,}")
        logger.info(f"Total features: {len(enhanced_df.columns)}")
        logger.info(f"Locations: {len(enhanced_df['location'].unique())}")
        logger.info(f"Date range: {enhanced_df['ds'].min()} to {enhanced_df['ds'].max()}")
        
        # Feature categories summary
        feature_categories = {
            'Holiday': len([col for col in enhanced_df.columns if 'holiday' in col or 'tet' in col]),
            'Weather/Seasonal': len([col for col in enhanced_df.columns if any(x in col for x in ['season', 'monsoon', 'temp', 'humidity', 'climate'])]),
            'Demographic': len([col for col in enhanced_df.columns if any(x in col for x in ['population', 'city', 'economic', 'healthcare'])]),
            'Temporal': len([col for col in enhanced_df.columns if any(x in col for x in ['month', 'quarter', 'week', 'day'])]),
            'Lagged/Rolling': len([col for col in enhanced_df.columns if 'lag' in col or 'rolling' in col or 'ema' in col]),
            'Fourier': len([col for col in enhanced_df.columns if 'sin' in col or 'cos' in col])
        }
        
        for category, count in feature_categories.items():
            logger.info(f"{category} features: {count}")
        
        logger.info("✅ Advanced feature engineering completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Advanced feature engineering failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

