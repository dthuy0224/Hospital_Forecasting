#!/usr/bin/env python3
"""
Prophet forecasting model for Hospital Demand Prediction
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import json
import logging
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HospitalDemandForecaster:
    """Prophet-based forecasting for hospital demand"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self.load_config(config_path)
        self.models = {}
        self.forecasts = {}
        self.performance_metrics = {}
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning("Config file not found, using defaults")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            'models': {
                'prophet': {
                    'seasonality_mode': 'multiplicative',
                    'yearly_seasonality': True,
                    'weekly_seasonality': True,
                    'daily_seasonality': False
                }
            },
            'forecasting': {
                'prediction_days': 14,
                'confidence_interval': 0.95
            }
        }
    
    def load_data(self, data_path: str = "data/processed/combined_forecast_data.csv") -> pd.DataFrame:
        """Load preprocessed data"""
        logger.info(f"📊 Loading data from {data_path}...")
        
        try:
            df = pd.read_csv(data_path)
            df['ds'] = pd.to_datetime(df['ds'])
            
            logger.info(f"✅ Loaded {len(df)} records for {df['location'].nunique()} locations")
            return df
        except FileNotFoundError:
            logger.warning(f"Data file not found: {data_path}")
            return pd.DataFrame()
    
    def prepare_prophet_data(self, df: pd.DataFrame, location: str) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        location_data = df[df['location'] == location].copy()
        location_data = location_data.sort_values('ds')
        
        # Ensure we have the required columns
        required_cols = ['ds', 'y']
        if not all(col in location_data.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Remove any rows with missing values in key columns
        location_data = location_data.dropna(subset=['ds', 'y'])
        
        return location_data
    
    def create_prophet_model(self, location: str) -> Prophet:
        """Create and configure Prophet model"""
        model_config = self.config.get('models', {}).get('prophet', {})
        
        model = Prophet(
            seasonality_mode=model_config.get('seasonality_mode', 'multiplicative'),
            yearly_seasonality=model_config.get('yearly_seasonality', True),
            weekly_seasonality=model_config.get('weekly_seasonality', True),
            daily_seasonality=model_config.get('daily_seasonality', False),
            interval_width=self.config.get('forecasting', {}).get('confidence_interval', 0.95)
        )
        
        # Add custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        
        # Add regressors if available
        model.add_regressor('is_weekend')
        model.add_regressor('month')
        
        logger.info(f"✅ Created Prophet model for {location}")
        return model
    
    def train_model(self, data: pd.DataFrame, location: str) -> Prophet:
        """Train Prophet model for a specific location"""
        logger.info(f"🎯 Training model for {location}...")
        
        # Prepare data
        train_data = self.prepare_prophet_data(data, location)
        
        if len(train_data) < 30:
            logger.warning(f"⚠️ Insufficient data for {location}: {len(train_data)} records")
            return None
        
        # Create and train model
        model = self.create_prophet_model(location)
        
        try:
            model.fit(train_data)
            self.models[location] = model
            logger.info(f"✅ Model trained for {location} with {len(train_data)} records")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to train model for {location}: {str(e)}")
            return None
    
    def generate_forecast(self, location: str, periods: int = None) -> pd.DataFrame:
        """Generate forecast for specific location"""
        if location not in self.models or self.models[location] is None:
            logger.error(f"❌ No trained model available for {location}")
            return pd.DataFrame()
        
        if periods is None:
            periods = self.config.get('forecasting', {}).get('prediction_days', 14)
        
        logger.info(f"🔮 Generating {periods}-day forecast for {location}...")
        
        model = self.models[location]
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Add regressor values for future dates
        future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        future['month'] = future['ds'].dt.month
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Store forecast
        self.forecasts[location] = forecast
        
        logger.info(f"✅ Generated forecast for {location}")
        return forecast
    
    def evaluate_model(self, data: pd.DataFrame, location: str, test_days: int = 30) -> Dict:
        """Evaluate model performance using holdout validation"""
        logger.info(f"📈 Evaluating model for {location}...")
        
        location_data = self.prepare_prophet_data(data, location)
        
        if len(location_data) < test_days + 30:
            logger.warning(f"⚠️ Insufficient data for evaluation: {len(location_data)} records")
            return {}
        
        # Split data
        train_data = location_data[:-test_days].copy()
        test_data = location_data[-test_days:].copy()
        
        # Train model on training data
        model = self.create_prophet_model(location)
        model.fit(train_data)
        
        # Generate forecast for test period
        future = model.make_future_dataframe(periods=test_days)
        future['is_weekend'] = future['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        future['month'] = future['ds'].dt.month
        
        forecast = model.predict(future)
        
        # Get predictions for test period
        test_predictions = forecast[-test_days:]['yhat'].values
        test_actual = test_data['y'].values
        
        # Calculate metrics
        mae = mean_absolute_error(test_actual, test_predictions)
        mse = mean_squared_error(test_actual, test_predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(test_actual, test_predictions) * 100
        
        # Accuracy (inverse of MAPE)
        accuracy = max(0, 100 - mape)
        
        metrics = {
            'location': location,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape),
            'accuracy': float(accuracy),
            'test_days': test_days,
            'mean_actual': float(np.mean(test_actual)),
            'mean_predicted': float(np.mean(test_predictions))
        }
        
        self.performance_metrics[location] = metrics
        
        logger.info(f"✅ Model evaluation for {location}: MAPE={mape:.2f}%, Accuracy={accuracy:.2f}%")
        
        return metrics
    
    def create_forecast_visualization(self, location: str) -> go.Figure:
        """Create interactive forecast visualization"""
        if location not in self.forecasts:
            logger.error(f"❌ No forecast available for {location}")
            return None
        
        forecast = self.forecasts[location]
        
        fig = go.Figure()
        
        # Historical data
        historical_end = len(forecast) - self.config.get('forecasting', {}).get('prediction_days', 14)
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'][:historical_end],
            y=forecast['yhat'][:historical_end],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast['ds'][historical_end-1:],
            y=forecast['yhat'][historical_end-1:],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'][historical_end-1:],
            y=forecast['yhat_upper'][historical_end-1:],
            mode='lines',
            name='Upper Bound',
            line=dict(color='gray', width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast['ds'][historical_end-1:],
            y=forecast['yhat_lower'][historical_end-1:],
            mode='lines',
            name='Lower Bound',
            line=dict(color='gray', width=0),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.2)',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f'Hospital Admission Forecast - {location}',
            xaxis_title='Date',
            yaxis_title='Daily Admissions',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def save_results(self):
        """Save models, forecasts, and metrics"""
        logger.info("💾 Saving results...")
        
        # Create directories if they don't exist
        import os
        os.makedirs('models/forecasts', exist_ok=True)
        
        # Save forecasts
        for location, forecast in self.forecasts.items():
            filename = f"models/forecasts/forecast_{location.replace(' ', '_').lower()}.csv"
            forecast.to_csv(filename, index=False)
        
        # Save performance metrics
        with open('models/performance_metrics.json', 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        
        # Save model summary
        summary = {
            'models_trained': len(self.models),
            'forecasts_generated': len(self.forecasts),
            'average_accuracy': np.mean([m['accuracy'] for m in self.performance_metrics.values()]) if self.performance_metrics else 0,
            'locations': list(self.models.keys()),
            'forecast_horizon': self.config.get('forecasting', {}).get('prediction_days', 14),
            'generated_at': datetime.now().isoformat()
        }
        
        with open('models/model_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("✅ Results saved")
    
    def train_all_models(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Train models for all locations"""
        logger.info("🎯 Training models for all locations...")
        
        if data.empty:
            logger.warning("⚠️ No data available for training")
            return {}
        
        locations = data['location'].unique()
        results = {}
        
        for location in locations:
            # Train model
            model = self.train_model(data, location)
            
            if model is not None:
                # Generate forecast
                forecast = self.generate_forecast(location)
                
                # Evaluate model
                metrics = self.evaluate_model(data, location)
                
                results[location] = {
                    'model_trained': True,
                    'forecast_generated': len(forecast) > 0,
                    'metrics': metrics
                }
            else:
                results[location] = {
                    'model_trained': False,
                    'forecast_generated': False,
                    'metrics': {}
                }
        
        logger.info(f"✅ Completed training for {len([r for r in results.values() if r['model_trained']])} locations")
        
        return results

def main():
    """Main function"""
    logger.info("🏥 Hospital Demand Forecasting with Prophet")
    logger.info("=" * 45)
    
    # Initialize forecaster
    forecaster = HospitalDemandForecaster()
    
    # Load data
    data = forecaster.load_data()
    
    if data.empty:
        logger.error("❌ No data available. Please run data processing first.")
        return
    
    # Train all models
    results = forecaster.train_all_models(data)
    
    # Save results
    forecaster.save_results()
    
    # Print summary
    successful_models = [loc for loc, result in results.items() if result['model_trained']]
    average_accuracy = np.mean([forecaster.performance_metrics[loc]['accuracy'] 
                              for loc in successful_models if loc in forecaster.performance_metrics])
    
    logger.info("\n📊 Training Summary:")
    logger.info(f"- Models trained: {len(successful_models)}/{len(results)}")
    logger.info(f"- Average accuracy: {average_accuracy:.2f}%")
    logger.info(f"- Forecast horizon: {forecaster.config.get('forecasting', {}).get('prediction_days', 14)} days")
    
    # Print top performing locations
    if forecaster.performance_metrics:
        sorted_metrics = sorted(forecaster.performance_metrics.items(), 
                              key=lambda x: x[1]['accuracy'], reverse=True)
        
        logger.info("\n🏆 Top 3 performing locations:")
        for location, metrics in sorted_metrics[:3]:
            logger.info(f"  {location}: {metrics['accuracy']:.2f}% accuracy (MAPE: {metrics['mape']:.2f}%)")
    
    logger.info("\n🎉 Forecasting completed!")
    logger.info("\n📋 Next steps:")
    logger.info("1. streamlit run src/visualization/streamlit_dashboard.py")

if __name__ == "__main__":
    main()