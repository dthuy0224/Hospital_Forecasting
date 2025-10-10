#!/usr/bin/env python3
"""
Simple Hyperparameter Optimization for Prophet Models
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data():
    """Load processed data"""
    print("📊 Loading data...")
    data_path = "data/processed/combined_forecast_data.csv"
    df = pd.read_csv(data_path)
    df['ds'] = pd.to_datetime(df['ds'])
    print(f"✅ Loaded {len(df)} records")
    return df

def optimize_location(location, df, max_tests=20):
    """Optimize hyperparameters for a single location"""
    print(f"\n🔍 Optimizing {location}...")
    
    # Prepare data
    location_data = df[df['location'] == location].copy()
    location_data = location_data.sort_values('ds').reset_index(drop=True)
    
    print(f"📈 Data points for {location}: {len(location_data)}")
    
    # Define parameter grid (smaller for efficiency)
    param_combinations = [
        # Conservative settings
        {
            'changepoint_prior_scale': 0.01,
            'seasonality_prior_scale': 0.1,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True
        },
        # Default Prophet settings
        {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True
        },
        # More flexible
        {
            'changepoint_prior_scale': 0.1,
            'seasonality_prior_scale': 1.0,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True
        },
        # Very flexible
        {
            'changepoint_prior_scale': 0.5,
            'seasonality_prior_scale': 0.1,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True
        },
        # Multiplicative seasonality
        {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative',
            'yearly_seasonality': True,
            'weekly_seasonality': True
        },
        # No yearly seasonality
        {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive',
            'yearly_seasonality': False,
            'weekly_seasonality': True
        },
        # No weekly seasonality
        {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': False
        }
    ]
    
    best_score = float('inf')
    best_params = None
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"  Testing combination {i+1}/{len(param_combinations)}...")
        
        try:
            # Create model
            model = Prophet(
                changepoint_prior_scale=params['changepoint_prior_scale'],
                seasonality_prior_scale=params['seasonality_prior_scale'],
                seasonality_mode=params['seasonality_mode'],
                yearly_seasonality=params['yearly_seasonality'],
                weekly_seasonality=params['weekly_seasonality'],
                daily_seasonality=False,
                interval_width=0.95
            )
            
            # Add regressors if available
            if 'is_weekend' in location_data.columns:
                model.add_regressor('is_weekend')
            if 'month' in location_data.columns:
                model.add_regressor('month')
            
            # Fit model
            model.fit(location_data)
            
            # Cross-validation (shorter for speed)
            cv_results = cross_validation(
                model, 
                initial='45 days',  # Shorter initial period
                period='7 days',
                horizon='14 days',
                parallel='threads'
            )
            
            # Calculate metrics
            metrics = performance_metrics(cv_results)
            mape = float(metrics['mape'].mean())
            
            print(f"    MAPE: {mape:.3f}")
            
            results.append({
                'params': params,
                'mape': mape,
                'mae': float(metrics['mae'].mean()),
                'rmse': float(metrics['rmse'].mean())
            })
            
            # Update best if improved
            if mape < best_score:
                best_score = mape
                best_params = params.copy()
                print(f"    🎯 New best MAPE: {mape:.3f}")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            continue
    
    print(f"✅ Best MAPE for {location}: {best_score:.3f}")
    print(f"   Best params: {best_params}")
    
    return {
        'location': location,
        'best_params': best_params,
        'best_mape': best_score,
        'all_results': results
    }

def main():
    """Main optimization function"""
    print("🚀 Starting Simple Hyperparameter Optimization")
    print("="*60)
    
    try:
        # Load data
        df = load_data()
        
        # Get locations
        locations = df['location'].unique()
        print(f"📍 Locations to optimize: {locations}")
        
        # Optimize each location
        all_results = {}
        for location in locations:
            result = optimize_location(location, df)
            all_results[location] = result
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"models/optimization_results_{timestamp}.json"
        os.makedirs("models", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to {results_file}")
        
        # Compare with current performance
        compare_with_current(all_results)
        
        print("\n🎉 Optimization completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def compare_with_current(optimized_results):
    """Compare with current performance"""
    print("\n📊 PERFORMANCE COMPARISON")
    print("="*60)
    
    # Load current metrics
    current_file = "models/performance_metrics.json"
    if not os.path.exists(current_file):
        print("⚠️ Current performance metrics not found")
        return
    
    with open(current_file, 'r') as f:
        current_metrics = json.load(f)
    
    print(f"{'Location':<20} {'Current MAPE':<15} {'Optimized MAPE':<16} {'Improvement':<12}")
    print("-"*60)
    
    total_improvement = 0
    improved_count = 0
    
    for location in optimized_results.keys():
        if location in current_metrics:
            current_mape = current_metrics[location]['mape']
            optimized_mape = optimized_results[location]['best_mape']
            improvement = ((current_mape - optimized_mape) / current_mape) * 100
            
            print(f"{location:<20} {current_mape:<15.3f} {optimized_mape:<16.3f} {improvement:<12.1f}%")
            
            if improvement > 0:
                improved_count += 1
                total_improvement += improvement
    
    print("-"*60)
    if improved_count > 0:
        avg_improvement = total_improvement / improved_count
        print(f"Average improvement: {avg_improvement:.1f}% across {improved_count} locations")
    else:
        print("No improvements found")
    print("="*60)

if __name__ == "__main__":
    main()
