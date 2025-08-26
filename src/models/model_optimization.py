#!/usr/bin/env python3
"""
Model Optimization Script for Hospital Forecasting Project
Implements hyperparameter tuning, cross-validation, and ensemble methods
"""

import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Advanced model optimization with hyperparameter tuning"""
    
    def __init__(self, data_path: str = "data/processed/combined_forecast_data.csv"):
        self.data_path = data_path
        self.optimized_models = {}
        self.cv_results = {}
        self.best_params = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare data for optimization"""
        logger.info("📊 Loading data for optimization...")
        
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
    
    def create_prophet_model(self, params: Dict[str, Any]) -> Prophet:
        """Create Prophet model with given parameters"""
        model = Prophet(
            changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
            holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
            seasonality_mode=params.get('seasonality_mode', 'additive'),
            changepoint_range=params.get('changepoint_range', 0.8),
            yearly_seasonality=params.get('yearly_seasonality', True),
            weekly_seasonality=params.get('weekly_seasonality', True),
            daily_seasonality=params.get('daily_seasonality', False)
        )
        
        # Add custom seasonalities if specified
        if params.get('add_custom_seasonality', False):
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        
        return model
    
    def evaluate_model_cv(self, model: Prophet, data: pd.DataFrame, 
                         initial: str = '180 days', period: str = '14 days', 
                         horizon: str = '7 days') -> Dict[str, float]:
        """Evaluate model using cross-validation"""
        try:
            # Perform cross-validation
            df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
            
            # Calculate performance metrics
            df_p = performance_metrics(df_cv)
            
            # Extract key metrics
            metrics = {
                'mape': df_p['mape'].mean(),
                'mae': df_p['mae'].mean(),
                'rmse': df_p['rmse'].mean(),
                'mdape': df_p['mdape'].mean(),
                'coverage': df_p['coverage'].mean()
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"⚠️ Cross-validation failed: {str(e)}")
            return {
                'mape': float('inf'),
                'mae': float('inf'),
                'rmse': float('inf'),
                'mdape': float('inf'),
                'coverage': 0.0
            }
    
    def grid_search_optimization(self, data: pd.DataFrame, location: str) -> Dict[str, Any]:
        """Perform grid search for hyperparameter optimization"""
        logger.info(f"🔍 Performing grid search for {location}...")
        
        # Define parameter grid (reduced for faster execution)
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1],
            'seasonality_prior_scale': [0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False]
        }
        
        # Generate all combinations
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        
        # Limit combinations for faster execution
        if len(all_params) > 20:
            np.random.seed(42)
            all_params = np.random.choice(all_params, 20, replace=False).tolist()
        
        best_score = float('inf')
        best_params = None
        best_metrics = None
        
        logger.info(f"🔍 Testing {len(all_params)} parameter combinations...")
        
        for i, params in enumerate(all_params):
            try:
                # Create and fit model
                model = self.create_prophet_model(params)
                model.fit(data)
                
                # Evaluate with cross-validation
                metrics = self.evaluate_model_cv(model, data)
                
                # Use MAPE as primary metric
                score = metrics['mape']
                
                if score < best_score and score < float('inf'):
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                
                if (i + 1) % 5 == 0:
                    logger.info(f"   Progress: {i+1}/{len(all_params)} - Best MAPE: {best_score:.2f}%")
                    
            except Exception as e:
                logger.debug(f"   Parameter set {i+1} failed: {str(e)}")
                continue
        
        if best_params is None:
            logger.warning(f"⚠️ No valid parameters found for {location}, using defaults")
            best_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10.0,
                'holidays_prior_scale': 10.0,
                'seasonality_mode': 'additive',
                'changepoint_range': 0.8,
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'add_custom_seasonality': False
            }
            best_metrics = {'mape': float('inf')}
        
        logger.info(f"✅ Best parameters for {location}: MAPE = {best_score:.2f}%")
        
        return {
            'best_params': best_params,
            'best_metrics': best_metrics,
            'best_score': best_score
        }
    
    def create_ensemble_model(self, data: pd.DataFrame, location: str) -> Dict[str, Any]:
        """Create ensemble model combining multiple Prophet configurations"""
        logger.info(f"🎯 Creating ensemble model for {location}...")
        
        # Define different model configurations
        ensemble_configs = [
            {
                'name': 'conservative',
                'params': {
                    'changepoint_prior_scale': 0.01,
                    'seasonality_prior_scale': 1.0,
                    'seasonality_mode': 'additive',
                    'yearly_seasonality': True,
                    'weekly_seasonality': True
                }
            },
            {
                'name': 'flexible',
                'params': {
                    'changepoint_prior_scale': 0.1,
                    'seasonality_prior_scale': 10.0,
                    'seasonality_mode': 'multiplicative',
                    'yearly_seasonality': True,
                    'weekly_seasonality': True
                }
            },
            {
                'name': 'smooth',
                'params': {
                    'changepoint_prior_scale': 0.001,
                    'seasonality_prior_scale': 0.1,
                    'seasonality_mode': 'additive',
                    'yearly_seasonality': True,
                    'weekly_seasonality': False
                }
            }
        ]
        
        ensemble_models = {}
        ensemble_forecasts = []
        
        # Train each model in ensemble
        for config in ensemble_configs:
            try:
                model = self.create_prophet_model(config['params'])
                model.fit(data)
                
                # Generate forecast
                future = model.make_future_dataframe(periods=14)
                forecast = model.predict(future)
                
                ensemble_models[config['name']] = model
                ensemble_forecasts.append(forecast[['ds', 'yhat']].set_index('ds'))
                
                logger.info(f"   ✅ Trained {config['name']} model")
                
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to train {config['name']} model: {str(e)}")
        
        if not ensemble_forecasts:
            logger.error(f"❌ No ensemble models could be trained for {location}")
            return None
        
        # Combine forecasts (simple average)
        combined_forecast = pd.concat(ensemble_forecasts, axis=1)
        combined_forecast.columns = [f'yhat_{i}' for i in range(len(ensemble_forecasts))]
        combined_forecast['yhat'] = combined_forecast.mean(axis=1)
        combined_forecast['yhat_std'] = combined_forecast.std(axis=1)
        
        # Calculate confidence intervals
        combined_forecast['yhat_lower'] = combined_forecast['yhat'] - 1.96 * combined_forecast['yhat_std']
        combined_forecast['yhat_upper'] = combined_forecast['yhat'] + 1.96 * combined_forecast['yhat_std']
        
        return {
            'models': ensemble_models,
            'forecast': combined_forecast.reset_index(),
            'configs': ensemble_configs
        }
    
    def optimize_all_models(self) -> Dict[str, Any]:
        """Optimize models for all locations"""
        logger.info("🚀 Starting model optimization for all locations...")
        
        # Load data
        data_by_location = self.load_data()
        
        optimization_results = {}
        
        for location, data in data_by_location.items():
            logger.info(f"\n🎯 Optimizing model for {location}...")
            
            try:
                # Grid search optimization
                grid_results = self.grid_search_optimization(data, location)
                
                # Create optimized model with best parameters
                best_model = self.create_prophet_model(grid_results['best_params'])
                best_model.fit(data)
                
                # Generate forecast with optimized model
                future = best_model.make_future_dataframe(periods=14)
                forecast = best_model.predict(future)
                
                # Create ensemble model
                ensemble_results = self.create_ensemble_model(data, location)
                
                # Compare single vs ensemble
                if ensemble_results:
                    # Calculate ensemble metrics on historical data
                    ensemble_historical = ensemble_results['forecast'][ensemble_results['forecast']['ds'] <= data['ds'].max()]
                    ensemble_mape = self.calculate_mape(data['y'].values, ensemble_historical['yhat'].values)
                    
                    # Choose better model
                    if ensemble_mape < grid_results['best_metrics']['mape']:
                        logger.info(f"   🏆 Ensemble model performs better: {ensemble_mape:.2f}% vs {grid_results['best_metrics']['mape']:.2f}%")
                        final_model = ensemble_results
                        final_metrics = {'mape': ensemble_mape}
                    else:
                        logger.info(f"   🏆 Single optimized model performs better: {grid_results['best_metrics']['mape']:.2f}% vs {ensemble_mape:.2f}%")
                        final_model = {
                            'type': 'single',
                            'model': best_model,
                            'forecast': forecast,
                            'params': grid_results['best_params']
                        }
                        final_metrics = grid_results['best_metrics']
                else:
                    final_model = {
                        'type': 'single',
                        'model': best_model,
                        'forecast': forecast,
                        'params': grid_results['best_params']
                    }
                    final_metrics = grid_results['best_metrics']
                
                optimization_results[location] = {
                    'model': final_model,
                    'metrics': final_metrics,
                    'grid_search': grid_results,
                    'data_points': len(data)
                }
                
                logger.info(f"   ✅ Optimization completed for {location}")
                
            except Exception as e:
                logger.error(f"   ❌ Optimization failed for {location}: {str(e)}")
                continue
        
        return optimization_results
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((actual - predicted) / actual)) * 100
    
    def save_optimized_models(self, results: Dict[str, Any]):
        """Save optimized models and results"""
        logger.info("💾 Saving optimized models...")
        
        # Create directories
        os.makedirs('models/optimized', exist_ok=True)
        os.makedirs('models/forecasts_optimized', exist_ok=True)
        
        # Save forecasts
        for location, result in results.items():
            forecast = result['model']['forecast']
            forecast_path = f"models/forecasts_optimized/forecast_{location.replace(' ', '_').lower()}.csv"
            forecast.to_csv(forecast_path, index=False)
            logger.info(f"   ✅ Saved forecast for {location}")
        
        # Save optimization results
        optimization_summary = {}
        for location, result in results.items():
            model_type = result['model'].get('type', 'ensemble' if 'models' in result['model'] else 'single')
            optimization_summary[location] = {
                'metrics': result['metrics'],
                'model_type': model_type,
                'data_points': result['data_points']
            }
        
        with open('models/optimization_results.json', 'w') as f:
            json.dump(optimization_summary, f, indent=2, default=str)
        
        logger.info("✅ All optimized models saved")
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        logger.info("📊 Generating optimization report...")
        
        report = {
            'optimization_date': datetime.now().isoformat(),
            'total_locations': len(results),
            'summary': {
                'average_mape': np.mean([r['metrics']['mape'] for r in results.values()]),
                'best_location': min(results.keys(), key=lambda x: results[x]['metrics']['mape']),
                'worst_location': max(results.keys(), key=lambda x: results[x]['metrics']['mape']),
                'models_improved': len([r for r in results.values() if r['metrics']['mape'] < 50])
            },
            'location_details': {}
        }
        
        for location, result in results.items():
            model_type = result['model'].get('type', 'ensemble' if 'models' in result['model'] else 'single')
            report['location_details'][location] = {
                'mape': result['metrics']['mape'],
                'model_type': model_type,
                'data_points': result['data_points'],
                'improvement_needed': result['metrics']['mape'] > 50
            }
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        with open('reports/optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("✅ Optimization report generated")
        return report

def main():
    """Main optimization function"""
    logger.info("🚀 Hospital Forecasting Model Optimization Started")
    logger.info("=" * 60)
    
    # Initialize optimizer
    optimizer = ModelOptimizer()
    
    try:
        # Perform optimization
        results = optimizer.optimize_all_models()
        
        if not results:
            logger.error("❌ No models were successfully optimized")
            return
        
        # Save optimized models
        optimizer.save_optimized_models(results)
        
        # Generate report
        report = optimizer.generate_optimization_report(results)
        
        # Print summary
        logger.info("\n📊 Optimization Summary:")
        logger.info(f"- Locations optimized: {report['total_locations']}")
        logger.info(f"- Average MAPE: {report['summary']['average_mape']:.2f}%")
        logger.info(f"- Best location: {report['summary']['best_location']}")
        logger.info(f"- Models with MAPE < 50%: {report['summary']['models_improved']}/{report['total_locations']}")
        
        logger.info("\n🎉 Model optimization completed!")
        logger.info("\n📋 Next steps:")
        logger.info("1. Review optimization_report.json")
        logger.info("2. Test optimized models with new data")
        logger.info("3. Deploy best performing models")
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 