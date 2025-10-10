#!/usr/bin/env python3
"""
Pytest configuration and fixtures for Hospital Forecasting tests
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta

@pytest.fixture
def sample_hospital_data():
    """Create sample hospital data for testing"""
    dates = pd.date_range('2023-01-01', periods=90, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'ds': dates,
        'y': np.random.randint(50, 200, 90),
        'location': ['Ho Chi Minh City'] * 90,
        'admission_count': np.random.randint(50, 200, 90),
        'disease_type': ['COVID-19'] * 90,
        'is_weekend': (dates.dayofweek >= 5).astype(int),
        'month': dates.month
    })
    return data

@pytest.fixture
def sample_multi_location_data():
    """Create sample data with multiple locations"""
    dates = pd.date_range('2023-01-01', periods=60, freq='D')
    locations = ['Ho Chi Minh City', 'Ha Noi', 'Da Nang']
    
    data_list = []
    for location in locations:
        location_data = pd.DataFrame({
            'date': dates,
            'ds': dates,
            'y': np.random.randint(50, 250, 60),
            'location': [location] * 60,
            'admission_count': np.random.randint(50, 250, 60),
            'disease_type': ['COVID-19'] * 60,
            'is_weekend': (dates.dayofweek >= 5).astype(int),
            'month': dates.month
        })
        data_list.append(location_data)
    
    return pd.concat(data_list, ignore_index=True)

@pytest.fixture
def sample_metrics():
    """Create sample performance metrics"""
    return {
        'Ho Chi Minh City': {
            'mape': 8.5,
            'accuracy': 91.5,
            'mae': 15.2,
            'rmse': 18.7,
            'test_days': 30,
            'mean_actual': 150.0,
            'mean_predicted': 145.0
        },
        'Ha Noi': {
            'mape': 7.3,
            'accuracy': 92.7,
            'mae': 14.8,
            'rmse': 17.9,
            'test_days': 30,
            'mean_actual': 160.0,
            'mean_predicted': 155.0
        },
        'Da Nang': {
            'mape': 7.9,
            'accuracy': 92.1,
            'mae': 15.5,
            'rmse': 19.2,
            'test_days': 30,
            'mean_actual': 140.0,
            'mean_predicted': 135.0
        }
    }

@pytest.fixture
def sample_forecast_data():
    """Create sample forecast data"""
    dates = pd.date_range('2023-04-01', periods=14, freq='D')
    return pd.DataFrame({
        'ds': dates,
        'yhat': np.random.randint(50, 200, 14),
        'yhat_lower': np.random.randint(30, 150, 14),
        'yhat_upper': np.random.randint(100, 250, 14),
        'location': ['Ho Chi Minh City'] * 14
    })

@pytest.fixture
def temp_csv_file(sample_hospital_data):
    """Create temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_hospital_data.to_csv(f.name, index=False)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)

@pytest.fixture
def temp_json_file(sample_metrics):
    """Create temporary JSON file for testing"""
    import json
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_metrics, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)

@pytest.fixture
def mock_config():
    """Create mock configuration for testing"""
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
        },
        'locations': [
            'Ho Chi Minh City',
            'Ha Noi',
            'Da Nang'
        ]
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker in item.keywords for marker in ['unit', 'integration']):
            item.add_marker(pytest.mark.unit)
