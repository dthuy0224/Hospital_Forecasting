#!/usr/bin/env python3
"""
Model Hyperparameter Optimization Script
Tối ưu hyperparameters cho Prophet models để cải thiện performance
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import Prophet and related modules
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelHyperparameterOptimizer:
    """Tối ưu hyperparameters cho Prophet models"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.results = {}
        
        # Hyperparameter search space
        self.param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_range': [0.8, 0.9, 0.95],
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False],
            'daily_seasonality': [False]  # Keep daily seasonality off for hospital data
        }
        
        # Best parameters found so far
        self.best_params = {}
        self.best_scores = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load processed data for optimization"""
        try:
            data_path = "data/processed/combined_forecast_data.csv"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            df = pd.read_csv(data_path)
            df['ds'] = pd.to_datetime(df['ds'])
            
            logger.info(f"✅ Loaded {len(df)} records from {data_path}")
            logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
            logger.info(f"Locations: {df['location'].unique()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise
    
    def prepare_data_for_location(self, df: pd.DataFrame, location: str) -> pd.DataFrame:
        """Prepare data for specific location"""
        location_data = df[df['location'] == location].copy()
        
        # Ensure we have required columns
        required_cols = ['ds', 'y']
        for col in required_cols:
            if col not in location_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add regressors if available
        regressor_cols = ['is_weekend', 'month']
        for col in regressor_cols:
            if col in location_data.columns:
                logger.info(f"✅ Using regressor: {col}")
        
        # Sort by date
        location_data = location_data.sort_values('ds').reset_index(drop=True)
        
        logger.info(f"✅ Prepared {len(location_data)} records for {location}")
        return location_data
    
    def create_model_with_params(self, params: Dict[str, Any], location_data: pd.DataFrame) -> Prophet:
        """Create Prophet model with given parameters"""
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            holidays_prior_scale=params['holidays_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            changepoint_range=params['changepoint_range'],
            yearly_seasonality=params['yearly_seasonality'],
            weekly_seasonality=params['weekly_seasonality'],
            daily_seasonality=params['daily_seasonality'],
            interval_width=0.95
        )
        
        # Add regressors if available
        regressor_cols = ['is_weekend', 'month']
        for col in regressor_cols:
            if col in location_data.columns:
                model.add_regressor(col)
        
        return model
    
    def evaluate_model(self, model: Prophet, location_data: pd.DataFrame, location: str) -> Dict[str, float]:
        """Evaluate model performance using cross-validation"""
        try:
            # Fit model
            model.fit(location_data)
            
            # Cross-validation
            cv_results = cross_validation(
                model, 
                initial='60 days',  # Use 60 days for initial training
                period='7 days',    # Evaluate every 7 days
                horizon='14 days',  # Forecast 14 days ahead
                parallel='threads'
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            # Return key metrics
            return {
                'mape': float(metrics['mape'].mean()),
                'mae': float(metrics['mae'].mean()),
                'rmse': float(metrics['rmse'].mean()),
                'coverage': float(metrics['coverage'].mean())
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Error evaluating model for {location}: {e}")
            return {
                'mape': 999.0,  # High penalty for failed evaluation
                'mae': 999.0,
                'rmse': 999.0,
                'coverage': 0.0
            }
    
    def optimize_hyperparameters(self, location: str, max_combinations: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific location"""
        logger.info(f"🔍 Starting hyperparameter optimization for {location}")
        
        # Load and prepare data
        df = self.load_data()
        location_data = self.prepare_data_for_location(df, location)
        
        # Generate parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        # Create all combinations
        all_combinations = list(product(*param_values))
        
        # Limit combinations if too many
        if len(all_combinations) > max_combinations:
            logger.info(f"⚠️ Limiting to {max_combinations} combinations out of {len(all_combinations)}")
            # Use random sampling for efficiency
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            combinations_to_test = [all_combinations[i] for i in indices]
        else:
            combinations_to_test = all_combinations
        
        logger.info(f"🧪 Testing {len(combinations_to_test)} parameter combinations")
        
        best_score = float('inf')
        best_params = None
        results = []
        
        for i, combination in enumerate(combinations_to_test):
            # Create parameter dictionary
            params = dict(zip(param_names, combination))
            
            logger.info(f"Testing combination {i+1}/{len(combinations_to_test)}: {params}")
            
            try:
                # Create and evaluate model
                model = self.create_model_with_params(params, location_data)
                metrics = self.evaluate_model(model, location_data, location)
                
                # Use MAPE as primary metric (lower is better)
                score = metrics['mape']
                
                # Store results
                result = {
                    'params': params,
                    'metrics': metrics,
                    'score': score
                }
                results.append(result)
                
                # Update best if improved
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"🎯 New best score for {location}: {score:.3f} MAPE")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to test combination {i+1}: {e}")
                continue
        
        # Store results
        self.results[location] = {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }
        
        logger.info(f"✅ Optimization completed for {location}")
        logger.info(f"Best MAPE: {best_score:.3f}")
        logger.info(f"Best params: {best_params}")
        
        return best_params
    
    def optimize_all_locations(self, max_combinations: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters for all locations"""
        logger.info("🚀 Starting hyperparameter optimization for all locations")
        
        # Get unique locations
        df = self.load_data()
        locations = df['location'].unique()
        
        logger.info(f"📍 Found {len(locations)} locations: {locations}")
        
        all_best_params = {}
        
        for location in locations:
            try:
                best_params = self.optimize_hyperparameters(location, max_combinations)
                all_best_params[location] = best_params
            except Exception as e:
                logger.error(f"❌ Failed to optimize {location}: {e}")
                continue
        
        # Save results
        self.save_optimization_results()
        
        logger.info("🎉 Hyperparameter optimization completed for all locations")
        return all_best_params
    
    def save_optimization_results(self):
        """Save optimization results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = f"models/optimization_results_{timestamp}.json"
        os.makedirs("models", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"💾 Saved detailed results to {results_file}")
        
        # Save best parameters summary
        summary = {}
        for location, result in self.results.items():
            summary[location] = {
                'best_params': result['best_params'],
                'best_mape': result['best_score']
            }
        
        summary_file = f"models/best_hyperparameters_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"💾 Saved best parameters to {summary_file}")
        
        # Update main optimization results file
        with open("models/optimization_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info("💾 Updated models/optimization_results.json")
    
    def compare_with_current_performance(self):
        """Compare optimized performance with current performance"""
        logger.info("📊 Comparing optimized vs current performance")
        
        # Load current performance metrics
        current_metrics_file = "models/performance_metrics.json"
        if os.path.exists(current_metrics_file):
            with open(current_metrics_file, 'r') as f:
                current_metrics = json.load(f)
        else:
            logger.warning("⚠️ Current performance metrics not found")
            return
        
        # Compare results
        print("\n" + "="*80)
        print("📈 PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Location':<20} {'Current MAPE':<15} {'Optimized MAPE':<16} {'Improvement':<12}")
        print("-"*80)
        
        total_improvement = 0
        locations_improved = 0
        
        for location in current_metrics.keys():
            if location in self.results:
                current_mape = current_metrics[location]['mape']
                optimized_mape = self.results[location]['best_score']
                improvement = ((current_mape - optimized_mape) / current_mape) * 100
                
                print(f"{location:<20} {current_mape:<15.3f} {optimized_mape:<16.3f} {improvement:<12.1f}%")
                
                if improvement > 0:
                    locations_improved += 1
                    total_improvement += improvement
        
        print("-"*80)
        if locations_improved > 0:
            avg_improvement = total_improvement / locations_improved
            print(f"Average improvement: {avg_improvement:.1f}% across {locations_improved} locations")
        else:
            print("No improvements found")
        print("="*80)

def main():
    """Main optimization function"""
    logger.info("🎯 Starting Model Hyperparameter Optimization")
    
    try:
        # Initialize optimizer
        optimizer = ModelHyperparameterOptimizer()
        
        # Run optimization for all locations
        best_params = optimizer.optimize_all_locations(max_combinations=30)  # Limit for efficiency
        
        # Compare with current performance
        optimizer.compare_with_current_performance()
        
        logger.info("🎉 Hyperparameter optimization completed successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("🏆 OPTIMIZATION SUMMARY")
        print("="*60)
        for location, params in best_params.items():
            score = optimizer.results[location]['best_score']
            print(f"{location}: {score:.3f} MAPE")
            print(f"  Best params: {params}")
            print()
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise

if __name__ == "__main__":
    main()
