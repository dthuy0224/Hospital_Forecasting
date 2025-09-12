#!/usr/bin/env python3
"""
Model Comparison Script for Hospital Forecasting Project
Compares Prophet and ARIMA models to find the best performing model
"""

import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Prophet imports
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# ARIMA imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logging.warning("⚠️ statsmodels not available, ARIMA models will be skipped")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelComparator:
    """Compare different forecasting models"""
    
    def __init__(self, data_path: str = "data/processed/combined_forecast_data.csv"):
        self.data_path = data_path
        self.comparison_results = {}
        self.best_models = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare data for comparison"""
        logger.info("📊 Loading data for model comparison...")
        
        try:
            df = pd.read_csv(self.data_path)
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Group by location
            data_by_location = {}
            for location in df['location'].unique():
                location_data = df[df['location'] == location].copy()
                location_data = location_data.sort_values('ds')
                data_by_location[location] = location_data[['ds', 'y']]
            
            logger.info(f"✅ Loaded data for {len(data_by_location)} locations")
            return data_by_location
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def train_prophet_model(self, data: pd.DataFrame, location: str) -> Dict[str, Any]:
        """Train Prophet model with optimized parameters"""
        logger.info(f"🤖 Training Prophet model for {location}...")
        
        try:
            # Use optimized parameters from previous optimization
            model = Prophet(
                changepoint_prior_scale=0.01,
                seasonality_prior_scale=1.0,
                holidays_prior_scale=1.0,
                seasonality_mode='additive',
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            
            model.fit(data)
            
            # Generate forecast
            future = model.make_future_dataframe(periods=14)
            forecast = model.predict(future)
            
            # Evaluate on historical data
            historical_forecast = forecast[forecast['ds'] <= data['ds'].max()]
            mape = self.calculate_mape(data['y'].values, historical_forecast['yhat'].values)
            
            return {
                'model': model,
                'forecast': forecast,
                'mape': mape,
                'type': 'prophet'
            }
            
        except Exception as e:
            logger.error(f"❌ Prophet training failed for {location}: {str(e)}")
            return None
    
    def train_arima_model(self, data: pd.DataFrame, location: str) -> Dict[str, Any]:
        """Train ARIMA model"""
        if not ARIMA_AVAILABLE:
            logger.warning("⚠️ ARIMA not available, skipping...")
            return None
        
        logger.info(f"📈 Training ARIMA model for {location}...")
        
        try:
            # Prepare time series data
            ts_data = data.set_index('ds')['y']
            
            # Check stationarity
            adf_result = adfuller(ts_data)
            is_stationary = adf_result[1] < 0.05
            
            if not is_stationary:
                # Make stationary by differencing
                ts_data = ts_data.diff().dropna()
            
            # Try different ARIMA parameters
            best_mape = float('inf')
            best_model = None
            best_forecast = None
            best_params = None
            
            # Grid search for ARIMA parameters (simplified)
            p_values = range(0, 2)
            d_values = range(0, 2)
            q_values = range(0, 2)
    
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            model = ARIMA(ts_data, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            # Generate forecast
                            forecast_steps = 14
                            forecast = fitted_model.forecast(steps=forecast_steps)
                            
                            # Calculate MAPE on historical data
                            historical_forecast = fitted_model.fittedvalues
                            if len(historical_forecast) > 0:
                                mape = self.calculate_mape(ts_data.values, historical_forecast.values)
                                
                                if mape < best_mape:
                                    best_mape = mape
                                    best_model = fitted_model
                                    best_forecast = forecast
                                    best_params = (p, d, q)
                                    
                        except Exception:
                            continue
            
            if best_model is None:
                logger.warning(f"⚠️ No valid ARIMA model found for {location}")
                return None
            
            # Create forecast dataframe
            last_date = data['ds'].max()
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=14, freq='D')
            
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': best_forecast.values
            })
            
            return {
                'model': best_model,
                'forecast': forecast_df,
                'mape': best_mape,
                'type': 'arima',
                'params': best_params
            }
            
        except Exception as e:
            logger.error(f"❌ ARIMA training failed for {location}: {str(e)}")
            return None
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((actual - predicted) / actual)) * 100
    
    def compare_models_for_location(self, data: pd.DataFrame, location: str) -> Dict[str, Any]:
        """Compare all models for a specific location"""
        logger.info(f"\n🔍 Comparing models for {location}...")
        
        results = {}
        
        # Train Prophet model
        prophet_result = self.train_prophet_model(data, location)
        if prophet_result:
            results['prophet'] = prophet_result
        
        # Train ARIMA model
        arima_result = self.train_arima_model(data, location)
        if arima_result:
            results['arima'] = arima_result
        
        # Find best model
        if results:
            best_model = min(results.keys(), key=lambda x: results[x]['mape'])
            best_mape = results[best_model]['mape']
            
            logger.info(f"   🏆 Best model for {location}: {best_model.upper()} (MAPE: {best_mape:.2f}%)")
            
            # Print all results
            for model_name, result in results.items():
                logger.info(f"   {model_name.upper()}: MAPE = {result['mape']:.2f}%")
        
        return results
    
    def compare_all_models(self) -> Dict[str, Any]:
        """Compare models for all locations"""
        logger.info("🚀 Starting model comparison for all locations...")
        
        # Load data
        data_by_location = self.load_data()
        
        comparison_results = {}
        
        for location, data in data_by_location.items():
            try:
                results = self.compare_models_for_location(data, location)
                comparison_results[location] = results
                
            except Exception as e:
                logger.error(f"❌ Model comparison failed for {location}: {str(e)}")
                continue
        
        return comparison_results
    
    def save_comparison_results(self, results: Dict[str, Any]):
        """Save comparison results"""
        logger.info("💾 Saving comparison results...")
        
        # Create directories
        os.makedirs('models/comparison', exist_ok=True)
        os.makedirs('models/forecasts_comparison', exist_ok=True)
        
        # Save forecasts for best models
        for location, location_results in results.items():
            if location_results:
                # Find best model
                best_model = min(location_results.keys(), key=lambda x: location_results[x]['mape'])
                best_result = location_results[best_model]
                
                # Save forecast
                forecast_path = f"models/forecasts_comparison/forecast_{location.replace(' ', '_').lower()}.csv"
                best_result['forecast'].to_csv(forecast_path, index=False)
                logger.info(f"   ✅ Saved best forecast for {location} ({best_model.upper()})")
        
        # Save comparison summary
        comparison_summary = {}
        for location, location_results in results.items():
            if location_results:
                comparison_summary[location] = {
                    'best_model': min(location_results.keys(), key=lambda x: location_results[x]['mape']),
                    'best_mape': location_results[min(location_results.keys(), key=lambda x: location_results[x]['mape'])]['mape'],
                    'all_models': {model: result['mape'] for model, result in location_results.items()}
                }
        
        with open('models/comparison_results.json', 'w') as f:
            json.dump(comparison_summary, f, indent=2, default=str)
        
        logger.info("✅ All comparison results saved")
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive comparison report"""
        logger.info("📊 Generating comparison report...")
        
        # Calculate statistics
        model_performance = {'prophet': [], 'arima': []}
        best_models_count = {'prophet': 0, 'arima': 0}
        
        for location, location_results in results.items():
            if location_results:
                # Track performance for each model
                for model_name in model_performance.keys():
                    if model_name in location_results:
                        model_performance[model_name].append(location_results[model_name]['mape'])
                
                # Count best models
                best_model = min(location_results.keys(), key=lambda x: location_results[x]['mape'])
                best_models_count[best_model] += 1
        
        # Calculate averages
        avg_performance = {}
        for model_name, performances in model_performance.items():
            if performances:
                avg_performance[model_name] = np.mean(performances)
            else:
                avg_performance[model_name] = float('inf')
        
        report = {
            'comparison_date': datetime.now().isoformat(),
            'total_locations': len(results),
            'models_compared': list(set([model for location_results in results.values() 
                                       for model in location_results.keys()])),
            'summary': {
                'average_performance': avg_performance,
                'best_model_overall': min(avg_performance.keys(), key=lambda x: avg_performance[x]),
                'best_models_count': best_models_count,
                'locations_with_improvement': len([r for r in results.values() 
                                                 if r and min(r.values(), key=lambda x: x['mape'])['mape'] < 50])
            },
            'location_details': {}
        }
        
        for location, location_results in results.items():
            if location_results:
                report['location_details'][location] = {
                    'best_model': min(location_results.keys(), key=lambda x: location_results[x]['mape']),
                    'best_mape': location_results[min(location_results.keys(), key=lambda x: location_results[x]['mape'])]['mape'],
                    'all_models_mape': {model: result['mape'] for model, result in location_results.items()}
                }
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        with open('reports/comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("✅ Comparison report generated")
        return report

def main():
    """Main comparison function"""
    logger.info("🚀 Hospital Forecasting Model Comparison Started")
    logger.info("=" * 60)
    
    # Initialize comparator
    comparator = ModelComparator()
    
    try:
        # Perform comparison
        results = comparator.compare_all_models()
        
        if not results:
            logger.error("❌ No models were successfully compared")
            return
        
        # Save comparison results
        comparator.save_comparison_results(results)
        
        # Generate report
        report = comparator.generate_comparison_report(results)
        
        # Print summary
        logger.info("\n📊 Comparison Summary:")
        logger.info(f"- Locations compared: {report['total_locations']}")
        logger.info(f"- Models compared: {', '.join(report['models_compared'])}")
        logger.info(f"- Best model overall: {report['summary']['best_model_overall'].upper()}")
        
        logger.info("\n🏆 Model Performance:")
        for model, avg_mape in report['summary']['average_performance'].items():
            if avg_mape < float('inf'):
                logger.info(f"   {model.upper()}: {avg_mape:.2f}% MAPE")
        
        logger.info("\n📈 Best Model Count:")
        for model, count in report['summary']['best_models_count'].items():
            logger.info(f"   {model.upper()}: {count} locations")
        
        logger.info("\n🎉 Model comparison completed!")
        logger.info("\n📋 Next steps:")
        logger.info("1. Review comparison_report.json")
        logger.info("2. Deploy best performing models")
        logger.info("3. Monitor model performance")
        
    except Exception as e:
        logger.error(f"❌ Model comparison failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 