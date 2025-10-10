import pandas as pd
import numpy as np
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import itertools
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedHyperparameterTuner:
    """Simplified hyperparameter tuning for Prophet models (sequential processing)"""
    
    def __init__(self, data_path: str = "data/processed/combined_forecast_data.csv"):
        self.data_path = data_path
        self.results = {}
        self.best_params = {}
        self.optimization_history = []
        
        # Simplified hyperparameter search space (reduced for faster execution)
        self.param_grid = {
            'seasonality_mode': ['additive', 'multiplicative'],
            'changepoint_prior_scale': [0.01, 0.05, 0.1],
            'seasonality_prior_scale': [0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.1, 1.0, 10.0],
            'n_changepoints': [15, 25, 35],
            'changepoint_range': [0.8, 0.9, 0.95],
            'yearly_seasonality': [True, False],
            'weekly_seasonality': [True, False],
            'daily_seasonality': [False],  # Keep False for daily data
            'interval_width': [0.80, 0.90, 0.95]
        }
        
        # Simplified seasonality configurations
        self.custom_seasonalities = [
            {'name': 'monthly', 'period': 30.5, 'fourier_order': [3, 5]},
            {'name': 'quarterly', 'period': 91.25, 'fourier_order': [3, 5]}
        ]
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load and prepare enhanced data for hyperparameter tuning"""
        logger.info("📊 Loading enhanced data for hyperparameter tuning...")
        
        try:
            df = pd.read_csv(self.data_path)
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Group by location
            data_by_location = {}
            for location in df['location'].unique():
                location_data = df[df['location'] == location].copy()
                location_data = location_data.sort_values('ds')
                
                # Prepare Prophet format with additional regressors
                prophet_data = location_data[['ds', 'y']].copy()
                
                # Add basic regressors (only if they exist in the data)
                basic_regressors = [
                    'is_weekend', 'is_holiday', 'month', 'quarter'
                ]
                
                for regressor in basic_regressors:
                    if regressor in location_data.columns:
                        prophet_data[regressor] = location_data[regressor]
                
                # Handle missing values
                prophet_data = prophet_data.fillna(method='ffill').fillna(method='bfill')
                
                data_by_location[location] = prophet_data
            
            logger.info(f"✅ Loaded data for {len(data_by_location)} locations")
            return data_by_location
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {str(e)}")
            raise
    
    def create_prophet_model_with_config(self, params: Dict[str, Any], 
                                       custom_seasonalities: List[Dict] = None) -> Prophet:
        """Create Prophet model with specific configuration"""
        try:
            # Create base model
            model = Prophet(
                seasonality_mode=params.get('seasonality_mode', 'multiplicative'),
                changepoint_prior_scale=params.get('changepoint_prior_scale', 0.05),
                seasonality_prior_scale=params.get('seasonality_prior_scale', 10.0),
                holidays_prior_scale=params.get('holidays_prior_scale', 10.0),
                n_changepoints=params.get('n_changepoints', 25),
                changepoint_range=params.get('changepoint_range', 0.8),
                yearly_seasonality=params.get('yearly_seasonality', True),
                weekly_seasonality=params.get('weekly_seasonality', True),
                daily_seasonality=params.get('daily_seasonality', False),
                interval_width=params.get('interval_width', 0.80)
            )
            
            # Add custom seasonalities if provided
            if custom_seasonalities:
                for seasonality in custom_seasonalities:
                    model.add_seasonality(
                        name=seasonality['name'],
                        period=seasonality['period'],
                        fourier_order=seasonality['fourier_order']
                    )
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Error creating Prophet model: {str(e)}")
            raise
    
    def evaluate_single_configuration(self, location: str, data: pd.DataFrame, 
                                    params: Dict[str, Any], seasonality_config: List[Dict], 
                                    config_id: int) -> Dict[str, Any]:
        """Evaluate a single hyperparameter configuration"""
        try:
            # Create model with configuration
            model = self.create_prophet_model_with_config(params, seasonality_config)
            
            # Add regressors (limit to basic ones to avoid issues)
            basic_regressors = [
                'is_weekend', 'is_holiday', 'month', 'quarter'
            ]
            
            for regressor in basic_regressors:
                if regressor in data.columns:
                    model.add_regressor(regressor)
            
            # Fit model
            model.fit(data)
            
            # Perform cross-validation with simplified parameters
            cv_results = cross_validation(
                model, 
                initial='120 days',  # Reduced for faster execution
                period='14 days',    # Reduced
                horizon='7 days',    # Reduced
                parallel=None        # Disable parallel processing
            )
            
            # Calculate performance metrics
            performance = performance_metrics(cv_results)
            
            # Extract key metrics
            metrics = {
                'location': location,
                'config_id': config_id,
                'params': params,
                'seasonality_config': seasonality_config,
                'mape': float(performance['mape'].mean()),
                'mae': float(performance['mae'].mean()),
                'rmse': float(performance['rmse'].mean()),
                'mdape': float(performance['mdape'].mean()),
                'coverage': float(performance['coverage'].mean()),
                'cv_folds': len(performance),
                'evaluation_time': time.time()
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"⚠️ Configuration {config_id} failed for {location}: {str(e)}")
            return {
                'location': location,
                'config_id': config_id,
                'params': params,
                'error': str(e),
                'mape': float('inf')
            }
    
    def generate_parameter_combinations(self, max_combinations: int = 50) -> List[Tuple]:
        """Generate parameter combinations for grid search"""
        logger.info(f"🔧 Generating parameter combinations (max: {max_combinations})...")
        try:
            # Generate base parameter combinations
            param_combinations = list(ParameterGrid(self.param_grid))
            
            # Generate seasonality combinations
            seasonality_combinations = []
            
            # No custom seasonality
            seasonality_combinations.append([])
            
            # Single seasonalities
            for seasonality in self.custom_seasonalities:
                for fourier_order in seasonality['fourier_order']:
                    config = [{
                        'name': seasonality['name'],
                        'period': seasonality['period'],
                        'fourier_order': fourier_order
                    }]
                    seasonality_combinations.append(config)
            
            # Combinations of seasonalities
            for combo in itertools.combinations(self.custom_seasonalities, 2):
                for fo1 in combo[0]['fourier_order']:
                    for fo2 in combo[1]['fourier_order']:
                        config = [
                            {
                                'name': combo[0]['name'],
                                'period': combo[0]['period'],
                                'fourier_order': fo1
                            },
                            {
                                'name': combo[1]['name'],
                                'period': combo[1]['period'],
                                'fourier_order': fo2
                            }
                        ]
                        seasonality_combinations.append(config)
            
            # Combine parameters and seasonalities
            all_combinations = []
            config_id = 0
            
            for params in param_combinations:
                for seasonality_config in seasonality_combinations:
                    all_combinations.append((params, seasonality_config, config_id))
                    config_id += 1
                    
                    if len(all_combinations) >= max_combinations:
                        break
                        
                if len(all_combinations) >= max_combinations:
                    break
            
            logger.info(f"✅ Generated {len(all_combinations)} parameter combinations")
            return all_combinations
            
        except Exception as e:
            logger.error(f"❌ Error generating parameter combinations: {str(e)}")
            raise
    
    def run_sequential_optimization(self) -> Dict[str, Any]:
        """Run sequential hyperparameter optimization"""
        logger.info("🚀 Starting sequential hyperparameter optimization...")
        
        try:
            # Load data
            data_by_location = self.load_data()
            
            # Generate parameter combinations
            param_combinations = self.generate_parameter_combinations()
            
            logger.info(f"💻 Running sequential optimization")
            
            # Prepare evaluation tasks
            evaluation_tasks = []
            for location, data in data_by_location.items():
                for params, seasonality_config, config_id in param_combinations:
                    task = (location, data, params, seasonality_config, config_id)
                    evaluation_tasks.append(task)
            
            logger.info(f"📋 Total evaluation tasks: {len(evaluation_tasks)}")
            
            # Run sequential evaluation
            results = []
            completed_tasks = 0
            start_time = time.time()
            
            for task in evaluation_tasks:
                location, data, params, seasonality_config, config_id = task
                
                try:
                    result = self.evaluate_single_configuration(
                        location, data, params, seasonality_config, config_id
                    )
                    results.append(result)
                    completed_tasks += 1
                    
                    # Progress reporting
                    if completed_tasks % 10 == 0 or completed_tasks == len(evaluation_tasks):
                        elapsed_time = time.time() - start_time
                        avg_time_per_task = elapsed_time / completed_tasks
                        remaining_tasks = len(evaluation_tasks) - completed_tasks
                        eta = remaining_tasks * avg_time_per_task
                        
                        logger.info(
                            f"📈 Progress: {completed_tasks}/{len(evaluation_tasks)} "
                            f"({completed_tasks/len(evaluation_tasks)*100:.1f}%) "
                            f"ETA: {eta/60:.1f}m"
                        )
                        
                except Exception as e:
                    logger.error(f"❌ Task failed: {str(e)}")
                    results.append({
                        'location': location,
                        'config_id': config_id,
                        'params': params,
                        'error': str(e),
                        'mape': float('inf')
                    })
                    completed_tasks += 1
            
            # Process results
            self.results = self.process_optimization_results(results)
            
            total_time = time.time() - start_time
            logger.info(f"✅ Sequential optimization completed in {total_time/60:.1f} minutes")
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Sequential optimization failed: {str(e)}")
            raise
    
    def process_optimization_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Process and analyze optimization results"""
        logger.info("📊 Processing optimization results...")
        
        try:
            # Filter out failed evaluations
            valid_results = [r for r in results if 'error' not in r and np.isfinite(r['mape'])]
            failed_results = [r for r in results if 'error' in r or not np.isfinite(r['mape'])]
            
            logger.info(f"✅ Valid results: {len(valid_results)}")
            logger.info(f"❌ Failed results: {len(failed_results)}")
            
            if len(valid_results) == 0:
                raise ValueError("No valid results from optimization")
            
            # Convert to DataFrame for analysis
            results_df = pd.DataFrame(valid_results)
            
            # Find best parameters for each location
            best_params_by_location = {}
            for location in results_df['location'].unique():
                location_results = results_df[results_df['location'] == location]
                best_result = location_results.loc[location_results['mape'].idxmin()]
                best_params_by_location[location] = best_result.to_dict()
            
            # Overall statistics
            overall_stats = {
                'total_evaluations': len(results),
                'valid_evaluations': len(valid_results),
                'failed_evaluations': len(failed_results),
                'success_rate': len(valid_results) / len(results) * 100,
                'best_overall_mape': float(results_df['mape'].min()),
                'worst_overall_mape': float(results_df['mape'].max()),
                'mean_mape': float(results_df['mape'].mean()),
                'std_mape': float(results_df['mape'].std()),
                'optimization_date': datetime.now().isoformat()
            }
            
            # Parameter importance analysis
            param_importance = self.analyze_parameter_importance(results_df)
            
            # Compile final results
            processed_results = {
                'best_params_by_location': best_params_by_location,
                'overall_statistics': overall_stats,
                'parameter_importance': param_importance,
                'all_results': valid_results,
                'failed_results': failed_results
            }
            
            logger.info("✅ Results processing completed")
            return processed_results
            
        except Exception as e:
            logger.error(f"❌ Error processing results: {str(e)}")
            raise
    
    def analyze_parameter_importance(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the importance of different hyperparameters"""
        logger.info("🔍 Analyzing parameter importance...")
        
        try:
            importance_analysis = {}
            
            # Analyze each parameter
            for param_name in self.param_grid.keys():
                if param_name in results_df.columns:
                    continue  # Skip if not directly available
                
                # Extract parameter values from params dict
                param_values = []
                mape_values = []
                
                for _, row in results_df.iterrows():
                    if param_name in row['params']:
                        param_values.append(row['params'][param_name])
                        mape_values.append(row['mape'])
                
                if len(param_values) > 0:
                    # Calculate statistics for each parameter value
                    param_stats = {}
                    unique_values = list(set(param_values))
                    
                    for value in unique_values:
                        value_indices = [i for i, v in enumerate(param_values) if v == value]
                        value_mapes = [mape_values[i] for i in value_indices]
                        
                        param_stats[str(value)] = {
                            'count': len(value_mapes),
                            'mean_mape': float(np.mean(value_mapes)),
                            'std_mape': float(np.std(value_mapes)),
                            'min_mape': float(np.min(value_mapes))
                        }
                    
                    importance_analysis[param_name] = param_stats
            
            logger.info("✅ Parameter importance analysis completed")
            return importance_analysis
            
        except Exception as e:
            logger.error(f"❌ Error analyzing parameter importance: {str(e)}")
            return {}
    
    def save_optimization_results(self, results: Dict[str, Any]):
        """Save optimization results and analysis"""
        logger.info("💾 Saving optimization results...")
        
        try:
            # Create directories
            os.makedirs('models/hyperparameter_optimization', exist_ok=True)
            os.makedirs('reports', exist_ok=True)
            
            # Save detailed results
            with open('models/hyperparameter_optimization/detailed_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            # Save best parameters for each location
            best_params_summary = {}
            for location, best_result in results['best_params_by_location'].items():
                best_params_summary[location] = {
                    'best_mape': best_result['mape'],
                    'best_params': best_result['params'],
                    'seasonality_config': best_result['seasonality_config']
                }
            
            with open('models/hyperparameter_optimization/best_parameters.json', 'w', encoding='utf-8') as f:
                json.dump(best_params_summary, f, indent=2, ensure_ascii=False, default=str)
            
            # Generate optimization report
            report = self.generate_optimization_report(results)
            with open('reports/hyperparameter_optimization_report.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info("✅ Optimization results saved successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to save optimization results: {str(e)}")
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        logger.info("📋 Generating optimization report...")
        
        try:
            stats = results['overall_statistics']
            best_params = results['best_params_by_location']
            
            report = {
                'report_type': 'hyperparameter_optimization',
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'total_evaluations': stats['total_evaluations'],
                    'success_rate': f"{stats['success_rate']:.1f}%",
                    'best_overall_mape': f"{stats['best_overall_mape']:.4f}%",
                    'mean_mape': f"{stats['mean_mape']:.4f}%",
                    'optimization_improvement': 'Significant' if stats['best_overall_mape'] < 10 else 'Moderate'
                },
                'best_performing_locations': {},
                'parameter_insights': [],
                'recommendations': []
            }
            
            # Analyze best performing locations
            location_performances = {}
            for location, result in best_params.items():
                location_performances[location] = result['best_mape']
            
            # Sort by performance
            sorted_locations = sorted(location_performances.items(), key=lambda x: x[1])
            report['best_performing_locations'] = {
                'best': sorted_locations[0],
                'worst': sorted_locations[-1],
                'rankings': sorted_locations
            }
            
            # Parameter insights
            param_importance = results.get('parameter_importance', {})
            for param_name, param_stats in param_importance.items():
                if param_stats:
                    best_value = min(param_stats.items(), key=lambda x: x[1]['mean_mape'])
                    report['parameter_insights'].append({
                        'parameter': param_name,
                        'best_value': best_value[0],
                        'best_mape': f"{best_value[1]['mean_mape']:.4f}%"
                    })
            
            # Generate recommendations
            if stats['best_overall_mape'] < 5:
                report['recommendations'].append("Excellent optimization results - models are ready for production")
            elif stats['best_overall_mape'] < 10:
                report['recommendations'].append("Good optimization results - consider additional feature engineering")
            else:
                report['recommendations'].append("Moderate results - consider ensemble methods or different algorithms")
            
            if stats['success_rate'] < 80:
                report['recommendations'].append("Low success rate - review parameter ranges and data quality")
            
            logger.info("✅ Optimization report generated")
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate optimization report: {str(e)}")
            return {}

def main():
    """Main function to run sequential hyperparameter optimization"""
    logger.info("🏥 Hospital Forecasting Sequential Hyperparameter Optimization Started")
    logger.info("=" * 70)
    
    try:
        # Initialize optimizer
        optimizer = SimplifiedHyperparameterTuner()
        
        # Run sequential optimization
        results = optimizer.run_sequential_optimization()
        
        # Save results
        optimizer.save_optimization_results(results)
        
        # Print summary
        stats = results['overall_statistics']
        best_params = results['best_params_by_location']
        
        logger.info("📊 HYPERPARAMETER OPTIMIZATION SUMMARY:")
        logger.info(f"Total evaluations: {stats['total_evaluations']:,}")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        logger.info(f"Best overall MAPE: {stats['best_overall_mape']:.4f}%")
        logger.info(f"Mean MAPE: {stats['mean_mape']:.4f}%")
        
        logger.info("\n🏆 BEST PARAMETERS BY LOCATION:")
        for location, result in best_params.items():
            logger.info(f"{location}: MAPE = {result['best_mape']:.4f}%")
            
        logger.info("✅ Sequential hyperparameter optimization completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Sequential hyperparameter optimization failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()

