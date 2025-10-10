#!/usr/bin/env python3
"""
Unit tests for model modules
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
import sys
import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.prophet_forecasting import HospitalDemandForecaster
from models.model_comparison import ModelComparator
from models.model_optimization import ModelOptimizer

class TestHospitalDemandForecaster(unittest.TestCase):
    """Test cases for HospitalDemandForecaster"""
    
    def setUp(self):
        """Set up test data"""
        self.forecaster = HospitalDemandForecaster()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=90, freq='D')
        self.sample_data = pd.DataFrame({
            'ds': dates,
            'y': np.random.randint(50, 200, 90),
            'location': ['Ho Chi Minh City'] * 90,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'month': dates.month
        })
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = self.forecaster.load_config('config/config.yaml')
        self.assertIsInstance(config, dict)
        self.assertIn('models', config)
        self.assertIn('forecasting', config)
    
    def test_default_config(self):
        """Test default configuration"""
        default_config = self.forecaster.get_default_config()
        self.assertIsInstance(default_config, dict)
        self.assertIn('models', default_config)
        self.assertIn('forecasting', default_config)
    
    def test_data_preparation(self):
        """Test data preparation for Prophet"""
        prophet_data = self.forecaster.prepare_prophet_data(self.sample_data, 'Ho Chi Minh City')
        
        # Check required columns
        self.assertIn('ds', prophet_data.columns)
        self.assertIn('y', prophet_data.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(prophet_data['ds']))
        self.assertTrue(pd.api.types.is_numeric_dtype(prophet_data['y']))
        
        # Check data integrity
        self.assertFalse(prophet_data['ds'].isna().any())
        self.assertFalse(prophet_data['y'].isna().any())
    
    def test_prophet_model_creation(self):
        """Test Prophet model creation"""
        model = self.forecaster.create_prophet_model('Ho Chi Minh City')
        
        # Check that model is created
        self.assertIsNotNone(model)
        
        # Check model configuration
        self.assertEqual(model.seasonality_mode, 'multiplicative')
        self.assertTrue(model.yearly_seasonality)
        self.assertTrue(model.weekly_seasonality)
        self.assertFalse(model.daily_seasonality)
    
    @patch('prophet.Prophet')
    def test_model_training(self, mock_prophet):
        """Test model training"""
        # Mock Prophet model
        mock_model = MagicMock()
        mock_prophet.return_value = mock_model
        
        # Create forecaster with mocked model
        forecaster = HospitalDemandForecaster()
        
        # Train model
        forecaster.train_models(self.sample_data)
        
        # Verify that fit was called
        mock_model.fit.assert_called_once()
        
        # Verify that model was stored
        self.assertIn('Ho Chi Minh City', forecaster.models)
    
    def test_forecast_generation(self):
        """Test forecast generation"""
        # Create a simple mock forecast
        future_dates = pd.date_range('2023-04-01', periods=14, freq='D')
        mock_forecast = pd.DataFrame({
            'ds': future_dates,
            'yhat': np.random.randint(50, 200, 14),
            'yhat_lower': np.random.randint(30, 150, 14),
            'yhat_upper': np.random.randint(100, 250, 14)
        })
        
        with patch.object(self.forecaster, 'models', {'Ho Chi Minh City': MagicMock()}):
            self.forecaster.models['Ho Chi Minh City'].predict.return_value = mock_forecast
            
            forecast = self.forecaster.generate_forecasts(self.sample_data)
            
            # Check that forecast was generated
            self.assertIn('Ho Chi Minh City', forecast)
            self.assertIsInstance(forecast['Ho Chi Minh City'], pd.DataFrame)
            self.assertEqual(len(forecast['Ho Chi Minh City']), 104)  # 90 + 14
    
    def test_performance_evaluation(self):
        """Test performance evaluation"""
        # Create mock data for evaluation
        actual = np.array([100, 120, 110, 130, 115])
        predicted = np.array([105, 118, 108, 125, 120])
        
        metrics = self.forecaster.calculate_metrics(actual, predicted)
        
        # Check that metrics are calculated
        self.assertIn('mae', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('accuracy', metrics)
        
        # Check that metrics are reasonable
        self.assertGreater(metrics['accuracy'], 0)
        self.assertLess(metrics['mape'], 100)

class TestModelComparator(unittest.TestCase):
    """Test cases for ModelComparator"""
    
    def setUp(self):
        """Set up test data"""
        self.comparator = ModelComparator()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=60, freq='D')
        self.sample_data = pd.DataFrame({
            'ds': dates,
            'y': np.random.randint(50, 200, 60),
            'location': ['Ho Chi Minh City'] * 60
        })
    
    def test_model_comparison_setup(self):
        """Test model comparison setup"""
        self.assertIsInstance(self.comparator, ModelComparator)
    
    @patch('prophet.Prophet')
    def test_prophet_model_training(self, mock_prophet):
        """Test Prophet model training in comparator"""
        mock_model = MagicMock()
        mock_prophet.return_value = mock_model
        
        # Test Prophet training
        result = self.comparator.train_prophet_model(self.sample_data)
        
        # Verify that model was trained
        mock_model.fit.assert_called_once()
        self.assertIsNotNone(result)
    
    def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        actual = np.array([100, 120, 110, 130, 115])
        predicted = np.array([105, 118, 108, 125, 120])
        
        metrics = self.comparator.calculate_metrics(actual, predicted)
        
        # Check that metrics are calculated
        expected_metrics = ['mae', 'mse', 'rmse', 'mape', 'accuracy']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        # Create mock results
        prophet_metrics = {'mape': 10.5, 'accuracy': 89.5}
        arima_metrics = {'mape': 12.3, 'accuracy': 87.7}
        
        comparison = self.comparator.compare_models(prophet_metrics, arima_metrics)
        
        # Check that comparison was made
        self.assertIn('best_model', comparison)
        self.assertIn('performance_difference', comparison)
        
        # Prophet should win with better MAPE
        self.assertEqual(comparison['best_model'], 'Prophet')

class TestModelOptimizer(unittest.TestCase):
    """Test cases for ModelOptimizer"""
    
    def setUp(self):
        """Set up test data"""
        self.optimizer = ModelOptimizer()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=90, freq='D')
        self.sample_data = pd.DataFrame({
            'ds': dates,
            'y': np.random.randint(50, 200, 90),
            'location': ['Ho Chi Minh City'] * 90
        })
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        self.assertIsInstance(self.optimizer, ModelOptimizer)
    
    def test_hyperparameter_grid_creation(self):
        """Test hyperparameter grid creation"""
        grid = self.optimizer.create_hyperparameter_grid()
        
        # Check that grid is created
        self.assertIsInstance(grid, dict)
        self.assertIn('changepoint_prior_scale', grid)
        self.assertIn('seasonality_prior_scale', grid)
        
        # Check that grid contains lists of values
        for param, values in grid.items():
            self.assertIsInstance(values, list)
            self.assertGreater(len(values), 1)
    
    @patch('prophet.Prophet')
    def test_hyperparameter_optimization(self, mock_prophet):
        """Test hyperparameter optimization"""
        mock_model = MagicMock()
        mock_prophet.return_value = mock_model
        
        # Mock cross-validation results
        mock_cv_results = pd.DataFrame({
            'mape': [10.5, 9.8, 11.2, 9.5, 10.8]
        })
        
        with patch.object(self.optimizer, 'cross_validate_model', return_value=mock_cv_results):
            best_params = self.optimizer.optimize_hyperparameters(self.sample_data)
            
            # Check that optimization was performed
            self.assertIsInstance(best_params, dict)
            self.assertIn('mape', best_params)
    
    def test_cross_validation(self):
        """Test cross-validation functionality"""
        # Create mock model
        mock_model = MagicMock()
        
        # Mock cross-validation results
        mock_cv_results = pd.DataFrame({
            'mape': [10.5, 9.8, 11.2, 9.5, 10.8],
            'mae': [15.2, 14.8, 16.1, 14.5, 15.7]
        })
        
        mock_model.cross_validation.return_value = mock_cv_results
        
        cv_results = self.optimizer.cross_validate_model(mock_model, self.sample_data)
        
        # Check that cross-validation was performed
        mock_model.cross_validation.assert_called_once()
        self.assertIsInstance(cv_results, pd.DataFrame)

class TestModelsIntegration(unittest.TestCase):
    """Integration tests for models"""
    
    def setUp(self):
        """Set up test data"""
        self.forecaster = HospitalDemandForecaster()
        self.comparator = ModelComparator()
        self.optimizer = ModelOptimizer()
        
        # Create comprehensive test data
        dates = pd.date_range('2023-01-01', periods=120, freq='D')
        self.test_data = pd.DataFrame({
            'ds': dates,
            'y': np.random.randint(50, 250, 120),
            'location': ['Ho Chi Minh City'] * 120,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'month': dates.month
        })
    
    def test_end_to_end_modeling(self):
        """Test complete modeling pipeline"""
        # This test would run the complete pipeline if not mocked
        # For now, we'll test that the components work together
        
        # Test data preparation
        prophet_data = self.forecaster.prepare_prophet_data(self.test_data, 'Ho Chi Minh City')
        self.assertIsInstance(prophet_data, pd.DataFrame)
        
        # Test model creation
        model = self.forecaster.create_prophet_model('Ho Chi Minh City')
        self.assertIsNotNone(model)
        
        # Test metrics calculation
        actual = np.array([100, 120, 110, 130, 115])
        predicted = np.array([105, 118, 108, 125, 120])
        
        metrics = self.forecaster.calculate_metrics(actual, predicted)
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestHospitalDemandForecaster))
    test_suite.addTest(unittest.makeSuite(TestModelComparator))
    test_suite.addTest(unittest.makeSuite(TestModelOptimizer))
    test_suite.addTest(unittest.makeSuite(TestModelsIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Models Tests Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
