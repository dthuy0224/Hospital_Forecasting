#!/usr/bin/env python3
"""
Backtesting Script for Hospital Forecasting Project
Implements time series cross-validation and historical performance evaluation
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelBacktester:
    """Comprehensive backtesting for time series forecasting models"""
    
    def __init__(self, data_path: str = "data/processed/combined_forecast_data.csv"):
        self.data_path = data_path
        self.backtest_results = {}
        self.cv_results = {}
        self.historical_performance = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare data for backtesting"""
        logger.info("📊 Loading data for backtesting...")
        
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
    
    def create_prophet_model(self, params: Dict[str, Any] = None) -> Prophet:
        """Create Prophet model with parameters"""
        if params is None:
            params = {
                'seasonality_mode': 'multiplicative',
                'yearly_seasonality': True,
                'weekly_seasonality': True,
                'daily_seasonality': False,
                'changepoint_prior_scale': 0.05
            }
        
        model = Prophet(**params)
        
        # Add custom seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
        
        return model
    
    def time_series_cross_validation(self, data: pd.DataFrame, location: str) -> Dict[str, Any]:
        logger.info(f"🔄 Performing cross-validation for {location}...")
        
        try:
            # Create model
            model = self.create_prophet_model()
            model.fit(data)
            
            # Cross-validation parameters
            initial = '180 days'  # Initial training period
            period = '30 days'    # Period between cutoff dates
            horizon = '14 days'   # Forecast horizon
            
            # Perform cross-validation
            df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
            
            # Calculate performance metrics
            df_p = performance_metrics(df_cv)
            
            # Extract key metrics
            metrics = {
                'mape': float(df_p['mape'].mean()),
                'mae': float(df_p['mae'].mean()),
                'rmse': float(df_p['rmse'].mean()),
                'mdape': float(df_p['mdape'].mean()),
                'coverage': float(df_p['coverage'].mean()),
                'cv_folds': len(df_p),
                'initial_period': initial,
                'period': period,
                'horizon': horizon
            }
            
            logger.info(f"✅ Cross-validation completed for {location}: MAPE = {metrics['mape']:.2f}%")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Cross-validation failed for {location}: {str(e)}")
            return {}
    
    def rolling_window_backtest(self, data: pd.DataFrame, location: str, 
                              window_size: int = 180, step_size: int = 30) -> Dict[str, Any]:
        """Perform rolling window backtesting"""
        logger.info(f"🔄 Performing rolling window backtest for {location}...")
        
        try:
            if len(data) < window_size + 30:
                logger.warning(f"⚠️ Insufficient data for {location}: {len(data)} records")
                return {}
            
            predictions = []
            actuals = []
            dates = []
            
            # Rolling window
            for start_idx in range(0, len(data) - window_size - 14, step_size):
                end_idx = start_idx + window_size
                
                # Training data
                train_data = data.iloc[start_idx:end_idx].copy()
                
                # Test data (next 14 days)
                test_data = data.iloc[end_idx:end_idx + 14].copy()
                
                if len(test_data) < 14:
                    break
                
                # Train model
                model = self.create_prophet_model()
                model.fit(train_data)
                
                # Generate forecast
                future = model.make_future_dataframe(periods=14)
                forecast = model.predict(future)
                
                # Get predictions for test period
                test_predictions = forecast.iloc[-14:]['yhat'].values
                test_actuals = test_data['y'].values
                test_dates = test_data['ds'].values
                
                predictions.extend(test_predictions)
                actuals.extend(test_actuals)
                dates.extend(test_dates)
            
            if len(predictions) == 0:
                logger.warning(f"⚠️ No predictions generated for {location}")
                return {}
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mae = mean_absolute_error(actuals, predictions)
            mse = mean_squared_error(actuals, predictions)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(actuals, predictions) * 100
            
            # Calculate directional accuracy
            actual_changes = np.diff(actuals)
            predicted_changes = np.diff(predictions)
            directional_accuracy = np.mean(np.sign(actual_changes) == np.sign(predicted_changes)) * 100
            
            results = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'directional_accuracy': float(directional_accuracy),
                'total_predictions': len(predictions),
                'window_size': window_size,
                'step_size': step_size,
                'predictions': predictions.tolist(),
                'actuals': actuals.tolist(),
                'dates': [str(d) for d in dates]
            }
            
            logger.info(f"✅ Rolling backtest completed for {location}: MAPE = {mape:.2f}%")
            return results
            
        except Exception as e:
            logger.error(f"❌ Rolling backtest failed for {location}: {str(e)}")
            return {}
    
    def historical_performance_analysis(self, data: pd.DataFrame, location: str) -> Dict[str, Any]:
        """Analyze historical performance patterns"""
        logger.info(f"📈 Analyzing historical performance for {location}...")
        
        try:
            # Calculate rolling statistics
            data['rolling_mean_7d'] = data['y'].rolling(window=7).mean()
            data['rolling_std_7d'] = data['y'].rolling(window=7).std()
            data['rolling_mean_30d'] = data['y'].rolling(window=30).mean()
            
            # Seasonal decomposition
            data['month'] = data['ds'].dt.month
            data['day_of_week'] = data['ds'].dt.dayofweek
            data['quarter'] = data['ds'].dt.quarter
            
            # Monthly averages
            monthly_avg = data.groupby('month')['y'].mean().to_dict()
            
            # Day of week averages
            dow_avg = data.groupby('day_of_week')['y'].mean().to_dict()
            
            # Trend analysis
            data['trend'] = np.arange(len(data))
            trend_coef = np.polyfit(data['trend'], data['y'], 1)[0]
            
            # Volatility analysis
            volatility = data['y'].std() / data['y'].mean() * 100
            
            # Peak and trough analysis
            peak_month = max(monthly_avg, key=monthly_avg.get)
            trough_month = min(monthly_avg, key=monthly_avg.get)
            
            analysis = {
                'total_records': len(data),
                'date_range': {
                    'start': str(data['ds'].min()),
                    'end': str(data['ds'].max()),
                    'days': len(data)
                },
                'statistics': {
                    'mean': float(data['y'].mean()),
                    'std': float(data['y'].std()),
                    'min': float(data['y'].min()),
                    'max': float(data['y'].max()),
                    'volatility': float(volatility)
                },
                'seasonal_patterns': {
                    'monthly_averages': monthly_avg,
                    'day_of_week_averages': dow_avg,
                    'peak_month': int(peak_month),
                    'trough_month': int(trough_month)
                },
                'trend': {
                    'slope': float(trend_coef),
                    'trend_direction': 'increasing' if trend_coef > 0 else 'decreasing'
                }
            }
            
            logger.info(f"✅ Historical analysis completed for {location}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Historical analysis failed for {location}: {str(e)}")
            return {}
    
    def run_comprehensive_backtest(self) -> Dict[str, Any]:
        """Run comprehensive backtesting for all locations"""
        logger.info("🚀 Starting comprehensive backtesting...")
        
        try:
            # Load data
            data_by_location = self.load_data()
            
            results = {}
            
            for location, data in data_by_location.items():
                logger.info(f"📍 Processing {location}...")
                
                location_results = {}
                
                # 1. Time series cross-validation
                cv_metrics = self.time_series_cross_validation(data, location)
                location_results['cross_validation'] = cv_metrics
                
                # 2. Rolling window backtest
                rolling_metrics = self.rolling_window_backtest(data, location)
                location_results['rolling_backtest'] = rolling_metrics
                
                # 3. Historical performance analysis
                historical_analysis = self.historical_performance_analysis(data, location)
                location_results['historical_analysis'] = historical_analysis
                
                # 4. Overall performance summary
                if cv_metrics and rolling_metrics:
                    overall_metrics = {
                        'cv_mape': cv_metrics.get('mape', 0),
                        'rolling_mape': rolling_metrics.get('mape', 0),
                        'avg_mape': (cv_metrics.get('mape', 0) + rolling_metrics.get('mape', 0)) / 2,
                        'directional_accuracy': rolling_metrics.get('directional_accuracy', 0),
                        'total_predictions': rolling_metrics.get('total_predictions', 0)
                    }
                    location_results['overall_performance'] = overall_metrics
                
                results[location] = location_results
            
            self.backtest_results = results
            
            # Generate summary
            summary = self.generate_backtest_summary(results)
            
            logger.info("✅ Comprehensive backtesting completed")
            return results
            
        except Exception as e:
            logger.error(f"❌ Comprehensive backtesting failed: {str(e)}")
            raise
    
    def generate_backtest_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of backtesting results"""
        logger.info("📊 Generating backtest summary...")
        
        try:
            summary = {
                'backtest_date': datetime.now().isoformat(),
                'total_locations': len(results),
                'overall_metrics': {},
                'location_rankings': {},
                'recommendations': []
            }
            
            # Calculate overall metrics
            all_cv_mape = []
            all_rolling_mape = []
            all_directional_accuracy = []
            
            for location, location_results in results.items():
                cv_metrics = location_results.get('cross_validation', {})
                rolling_metrics = location_results.get('rolling_backtest', {})
                overall_metrics = location_results.get('overall_performance', {})
                
                if cv_metrics:
                    all_cv_mape.append(cv_metrics.get('mape', 0))
                if rolling_metrics:
                    all_rolling_mape.append(rolling_metrics.get('mape', 0))
                    all_directional_accuracy.append(rolling_metrics.get('directional_accuracy', 0))
                
                # Store location performance for ranking
                if overall_metrics:
                    summary['location_rankings'][location] = {
                        'avg_mape': overall_metrics.get('avg_mape', 0),
                        'directional_accuracy': overall_metrics.get('directional_accuracy', 0),
                        'total_predictions': overall_metrics.get('total_predictions', 0)
                    }
            
            # Overall statistics
            if all_cv_mape:
                summary['overall_metrics']['avg_cv_mape'] = float(np.mean(all_cv_mape))
                summary['overall_metrics']['best_cv_mape'] = float(np.min(all_cv_mape))
                summary['overall_metrics']['worst_cv_mape'] = float(np.max(all_cv_mape))
            
            if all_rolling_mape:
                summary['overall_metrics']['avg_rolling_mape'] = float(np.mean(all_rolling_mape))
                summary['overall_metrics']['best_rolling_mape'] = float(np.min(all_rolling_mape))
                summary['overall_metrics']['worst_rolling_mape'] = float(np.max(all_rolling_mape))
            
            if all_directional_accuracy:
                summary['overall_metrics']['avg_directional_accuracy'] = float(np.mean(all_directional_accuracy))
            
            # Generate recommendations
            if summary['overall_metrics'].get('avg_cv_mape', 0) > 20:
                summary['recommendations'].append("High MAPE detected - consider feature engineering or model tuning")
            
            if summary['overall_metrics'].get('avg_directional_accuracy', 0) < 50:
                summary['recommendations'].append("Low directional accuracy - consider trend analysis improvements")
            
            # Find best and worst performing locations
            if summary['location_rankings']:
                best_location = min(summary['location_rankings'].items(), 
                                  key=lambda x: x[1]['avg_mape'])
                worst_location = max(summary['location_rankings'].items(), 
                                   key=lambda x: x[1]['avg_mape'])
                
                summary['best_performing_location'] = best_location[0]
                summary['worst_performing_location'] = worst_location[0]
            
            logger.info("✅ Backtest summary generated")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate backtest summary: {str(e)}")
            return {}
    
    def save_backtest_results(self, results: Dict[str, Any]):
        """Save backtesting results to files"""
        logger.info("💾 Saving backtest results...")
        
        try:
            # Create directories
            os.makedirs('models/backtest', exist_ok=True)
            os.makedirs('reports', exist_ok=True)
            
            # Save detailed results
            with open('models/backtest/detailed_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            # Save summary
            summary = self.generate_backtest_summary(results)
            with open('reports/backtest_summary.json', 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("✅ Backtest results saved")
            
        except Exception as e:
            logger.error(f"❌ Failed to save backtest results: {str(e)}")
    
    def generate_backtest_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive backtesting report"""
        logger.info("📋 Generating backtest report...")
        
        try:
            summary = self.generate_backtest_summary(results)
            
            report = {
                'report_type': 'backtesting',
                'generated_at': datetime.now().isoformat(),
                'summary': summary,
                'detailed_results': results,
                'methodology': {
                    'cross_validation': {
                        'initial_period': '180 days',
                        'period': '30 days',
                        'horizon': '14 days'
                    },
                    'rolling_backtest': {
                        'window_size': 180,
                        'step_size': 30,
                        'forecast_horizon': 14
                    }
                },
                'key_findings': []
            }
            
            # Add key findings
            avg_mape = summary['overall_metrics'].get('avg_cv_mape', 0)
            if avg_mape < 10:
                report['key_findings'].append("Excellent model performance with MAPE < 10%")
            elif avg_mape < 15:
                report['key_findings'].append("Good model performance with MAPE < 15%")
            elif avg_mape < 20:
                report['key_findings'].append("Acceptable model performance with MAPE < 20%")
            else:
                report['key_findings'].append("Model performance needs improvement - MAPE > 20%")
            
            # Add location-specific findings
            for location, location_results in results.items():
                overall_metrics = location_results.get('overall_performance', {})
                if overall_metrics:
                    mape = overall_metrics.get('avg_mape', 0)
                    if mape < 10:
                        report['key_findings'].append(f"{location}: Excellent performance (MAPE: {mape:.2f}%)")
                    elif mape > 20:
                        report['key_findings'].append(f"{location}: Needs improvement (MAPE: {mape:.2f}%)")
            
            # Save report
            with open('reports/backtest_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("✅ Backtest report generated")
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate backtest report: {str(e)}")
            return {}

def main():
    """Main function to run backtesting"""
    logger.info("🏥 Hospital Forecasting Backtesting Started")
    logger.info("=" * 50)
    
    try:
        # Initialize backtester
        backtester = ModelBacktester()
        
        # Run comprehensive backtesting
        results = backtester.run_comprehensive_backtest()
        
        # Save results
        backtester.save_backtest_results(results)
        
        # Generate report
        report = backtester.generate_backtest_report(results)
        
        # Print summary
        summary = report.get('summary', {})
        logger.info("📊 BACKTESTING SUMMARY:")
        logger.info(f"Total locations: {summary.get('total_locations', 0)}")
        logger.info(f"Average CV MAPE: {summary.get('overall_metrics', {}).get('avg_cv_mape', 0):.2f}%")
        logger.info(f"Average Rolling MAPE: {summary.get('overall_metrics', {}).get('avg_rolling_mape', 0):.2f}%")
        logger.info(f"Best performing: {summary.get('best_performing_location', 'N/A')}")
        logger.info(f"Worst performing: {summary.get('worst_performing_location', 'N/A')}")
        
        logger.info("✅ Backtesting completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Backtesting failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 