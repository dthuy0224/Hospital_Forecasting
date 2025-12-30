#!/usr/bin/env python3
"""
XGBoost Forecasting Model for Hospital Demand Prediction
Uses gradient boosting with time series features for accurate predictions
"""

import pandas as pd
import numpy as np
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available for hyperparameter tuning. Install with: pip install optuna")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBoostForecaster:
    """XGBoost-based time series forecasting for hospital demand"""
    
    def __init__(self, data_path: str = "data/processed/enhanced_forecast_data.csv"):
        self.data_path = data_path
        self.models = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        self.best_params = {}
        
        # Default XGBoost parameters
        self.default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'random_state': 42
        }
        
        # Features to use (will be determined from data)
        self.feature_cols = []
        self.target_col = 'y'
        
    def load_data(self) -> pd.DataFrame:
        """Load enhanced dataset with advanced features"""
        logger.info(f"Loading data from {self.data_path}...")
        
        # Try enhanced data first, fall back to combined data
        if os.path.exists(self.data_path):
            df = pd.read_csv(self.data_path)
        else:
            fallback_path = "data/processed/combined_forecast_data.csv"
            logger.warning(f"Enhanced data not found, using {fallback_path}")
            df = pd.read_csv(fallback_path)
        
        df['ds'] = pd.to_datetime(df['ds'])
        
        logger.info(f"Loaded {len(df):,} records with {len(df.columns)} columns")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for XGBoost model"""
        logger.info("Preparing features for XGBoost...")
        
        # Create a copy
        df = df.copy()
        
        # Add basic time features if not present
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['ds'].dt.dayofweek
        if 'day_of_month' not in df.columns:
            df['day_of_month'] = df['ds'].dt.day
        if 'week_of_year' not in df.columns:
            df['week_of_year'] = df['ds'].dt.isocalendar().week
        if 'quarter' not in df.columns:
            df['quarter'] = df['ds'].dt.quarter
        
        # Create lag features if not present
        for lag in [1, 7, 14, 30]:
            lag_col = f'lag_{lag}'
            if lag_col not in df.columns:
                df[lag_col] = df.groupby('location')['y'].shift(lag)
        
        # Create rolling features if not present
        for window in [7, 14, 30]:
            mean_col = f'rolling_mean_{window}'
            std_col = f'rolling_std_{window}'
            if mean_col not in df.columns:
                df[mean_col] = df.groupby('location')['y'].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
            if std_col not in df.columns:
                df[std_col] = df.groupby('location')['y'].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
                )
        
        # Select feature columns (exclude non-features)
        exclude_cols = ['ds', 'y', 'location', 'created_at', 'date']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Only keep numeric columns
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # Fill NaN values
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        self.feature_cols = numeric_cols
        logger.info(f"Selected {len(numeric_cols)} features for training")
        
        return df, numeric_cols
    
    def train_model(self, data: pd.DataFrame, location: str, 
                    params: Dict = None, use_optuna: bool = False) -> Dict:
        """Train XGBoost model for a specific location"""
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not available")
            return {}
        
        logger.info(f"Training XGBoost model for {location}...")
        
        # Filter data for location
        location_data = data[data['location'] == location].copy()
        location_data = location_data.sort_values('ds')
        
        if len(location_data) < 60:
            logger.warning(f"Insufficient data for {location}: {len(location_data)} records")
            return {}
        
        # Prepare features
        location_data, feature_cols = self.prepare_features(location_data)
        
        # Split into train/test (last 30 days for testing)
        test_days = 30
        train_data = location_data[:-test_days]
        test_data = location_data[-test_days:]
        
        X_train = train_data[feature_cols]
        y_train = train_data[self.target_col]
        X_test = test_data[feature_cols]
        y_test = test_data[self.target_col]
        
        # Hyperparameter tuning with Optuna
        if use_optuna and OPTUNA_AVAILABLE:
            best_params = self._optimize_hyperparameters(X_train, y_train, location)
            params = {**self.default_params, **best_params}
        elif params is None:
            params = self.default_params
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Store model
        self.models[location] = model
        self.best_params[location] = params
        
        # Get feature importance
        importance = dict(zip(feature_cols, model.feature_importances_))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        self.feature_importance[location] = sorted_importance
        
        # Evaluate
        y_pred = model.predict(X_test)
        metrics = self._calculate_metrics(y_test.values, y_pred, location)
        
        logger.info(f"XGBoost model for {location}: MAPE={metrics['mape']:.2f}%, Accuracy={metrics['accuracy']:.2f}%")
        
        return metrics
    
    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                   location: str, n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Optuna"""
        logger.info(f"Optimizing hyperparameters for {location} with Optuna...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBRegressor(**params, random_state=42, objective='reg:squarederror')
                model.fit(X_train, y_train, verbose=False)
                
                y_pred = model.predict(X_val)
                mae = mean_absolute_error(y_val, y_pred)
                scores.append(mae)
            
            return np.mean(scores)
        
        # Create and run study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        logger.info(f"Best parameters found: {study.best_params}")
        
        return study.best_params
    
    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray, 
                           location: str) -> Dict:
        """Calculate performance metrics"""
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        # MAPE (avoid division by zero)
        non_zero_mask = actual > 0
        if non_zero_mask.sum() > 0:
            mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100
        else:
            mape = 0.0
        
        # SMAPE
        denominator = (np.abs(actual) + np.abs(predicted)) / 2
        denominator = np.where(denominator == 0, 1, denominator)
        smape = np.mean(np.abs(actual - predicted) / denominator) * 100
        
        # Accuracy
        mape_for_accuracy = mape if mape < 200 else smape
        accuracy = max(0, min(100, 100 - mape_for_accuracy))
        
        metrics = {
            'location': location,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(min(mape, 999)),
            'smape': float(smape),
            'accuracy': float(accuracy),
            'mean_actual': float(np.mean(actual)),
            'mean_predicted': float(np.mean(predicted)),
            'model': 'XGBoost'
        }
        
        self.performance_metrics[location] = metrics
        return metrics
    
    def generate_forecast(self, data: pd.DataFrame, location: str, 
                          periods: int = 14) -> pd.DataFrame:
        """Generate future forecast"""
        if location not in self.models:
            logger.error(f"No model found for {location}")
            return pd.DataFrame()
        
        logger.info(f"Generating {periods}-day forecast for {location}...")
        
        model = self.models[location]
        location_data = data[data['location'] == location].copy()
        location_data = location_data.sort_values('ds')
        
        # Prepare features
        location_data, feature_cols = self.prepare_features(location_data)
        
        # Get the last date
        last_date = location_data['ds'].max()
        
        # Generate future dates
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        # Create future dataframe with features
        future_df = pd.DataFrame({'ds': future_dates})
        future_df['location'] = location
        
        # Add time features
        future_df['day_of_week'] = future_df['ds'].dt.dayofweek
        future_df['day_of_month'] = future_df['ds'].dt.day
        future_df['month'] = future_df['ds'].dt.month
        future_df['week_of_year'] = future_df['ds'].dt.isocalendar().week
        future_df['quarter'] = future_df['ds'].dt.quarter
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        
        # For lag features, use last known values
        last_values = location_data[['y'] + [c for c in feature_cols if 'lag_' in c or 'rolling_' in c]].tail(1)
        for col in last_values.columns:
            if col != 'y':
                future_df[col] = last_values[col].values[0]
        
        # Fill missing features with 0
        for col in feature_cols:
            if col not in future_df.columns:
                future_df[col] = 0
        
        # Make predictions
        X_future = future_df[feature_cols].fillna(0)
        predictions = model.predict(X_future)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': predictions,
            'location': location
        })
        
        logger.info(f"Generated forecast for {location}: mean={predictions.mean():.2f}")
        
        return forecast_df
    
    def train_all_models(self, use_optuna: bool = False) -> Dict:
        """Train XGBoost models for all locations"""
        logger.info("Training XGBoost models for all locations...")
        
        # Load data
        data = self.load_data()
        locations = data['location'].unique()
        
        results = {}
        for location in locations:
            metrics = self.train_model(data, location, use_optuna=use_optuna)
            if metrics:
                results[location] = metrics
        
        # Summary
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
            avg_mape = np.mean([r['mape'] for r in results.values()])
            
            logger.info(f"\nXGBoost Training Summary:")
            logger.info(f"- Models trained: {len(results)}/{len(locations)}")
            logger.info(f"- Average accuracy: {avg_accuracy:.2f}%")
            logger.info(f"- Average MAPE: {avg_mape:.2f}%")
            
            # Save results
            self.save_results()
        
        return results
    
    def save_results(self):
        """Save model results and metrics"""
        os.makedirs('models', exist_ok=True)
        
        # Save performance metrics
        metrics_path = 'models/xgboost_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.performance_metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save feature importance (convert numpy types to Python types)
        importance_path = 'models/xgboost_feature_importance.json'
        importance_serializable = {
            loc: {k: float(v) for k, v in imp.items()} 
            for loc, imp in self.feature_importance.items()
        }
        with open(importance_path, 'w') as f:
            json.dump(importance_serializable, f, indent=2)
        logger.info(f"Saved feature importance to {importance_path}")
        
        # Save best parameters
        params_path = 'models/xgboost_best_params.json'
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        logger.info(f"Saved best parameters to {params_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='XGBoost Hospital Demand Forecasting')
    parser.add_argument('--data-path', default='data/processed/enhanced_forecast_data.csv',
                        help='Path to enhanced feature data')
    parser.add_argument('--use-optuna', action='store_true',
                        help='Use Optuna for hyperparameter optimization')
    parser.add_argument('--forecast-days', type=int, default=14,
                        help='Number of days to forecast')
    
    args = parser.parse_args()
    
    if not XGBOOST_AVAILABLE:
        print("XGBoost not installed. Please run: pip install xgboost")
        return
    
    # Initialize forecaster
    forecaster = XGBoostForecaster(args.data_path)
    
    # Train models
    results = forecaster.train_all_models(use_optuna=args.use_optuna)
    
    if results:
        # Generate forecasts
        data = forecaster.load_data()
        for location in results.keys():
            forecast = forecaster.generate_forecast(data, location, args.forecast_days)
            if not forecast.empty:
                forecast_path = f"models/forecasts/xgboost_{location.replace(' ', '_').lower()}_forecast.csv"
                os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
                forecast.to_csv(forecast_path, index=False)
        
        print("\n" + "="*50)
        print("XGBoost Training Completed!")
        print("="*50)
        for loc, metrics in results.items():
            print(f"{loc}: Accuracy={metrics['accuracy']:.2f}%, MAPE={metrics['mape']:.2f}%")


if __name__ == "__main__":
    main()
