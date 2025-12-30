#!/usr/bin/env python3
"""
FastAPI REST API for Hospital Demand Forecasting
Provides endpoints for predictions, metrics, and model information
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hospital Demand Forecasting API",
    description="REST API for hospital admission predictions using XGBoost and Prophet models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Pydantic Models ==============

class PredictionRequest(BaseModel):
    location: str = Field(..., description="Location name for prediction")
    days: int = Field(default=14, ge=1, le=90, description="Number of days to forecast")
    model: str = Field(default="xgboost", description="Model to use: 'xgboost' or 'prophet'")

class PredictionResponse(BaseModel):
    location: str
    model: str
    predictions: List[Dict[str, Any]]
    generated_at: str

class LocationInfo(BaseModel):
    name: str
    total_records: int
    date_range: Dict[str, str]
    avg_daily_admissions: float

class MetricsResponse(BaseModel):
    model: str
    metrics: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    data_available: bool


# ============== Data Loading ==============

def get_data_path() -> str:
    """Get the correct data path"""
    paths = [
        "data/processed/enhanced_forecast_data.csv",
        "data/processed/combined_forecast_data.csv"
    ]
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


def load_forecast_data() -> pd.DataFrame:
    """Load forecast data"""
    path = get_data_path()
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    return pd.DataFrame()


def load_metrics(model: str = "xgboost") -> Dict:
    """Load model metrics"""
    metrics_path = f"models/{model}_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return {}


def get_available_locations() -> List[str]:
    """Get list of available locations"""
    df = load_forecast_data()
    if not df.empty:
        return sorted(df['location'].unique().tolist())
    return []


# ============== Prediction Functions ==============

def generate_xgboost_forecast(location: str, days: int) -> List[Dict]:
    """Generate forecast using XGBoost model"""
    try:
        # Try to load saved forecast
        forecast_path = f"models/forecasts/xgboost_{location.replace(' ', '_').lower()}_forecast.csv"
        if os.path.exists(forecast_path):
            forecast_df = pd.read_csv(forecast_path)
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            
            predictions = []
            for _, row in forecast_df.head(days).iterrows():
                predictions.append({
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "predicted_admissions": max(0, round(row['yhat'], 2)),
                    "model": "xgboost"
                })
            return predictions
        
        # If no saved forecast, generate using loaded model
        # Import here to avoid circular imports
        from src.models.xgboost_forecasting import XGBoostForecaster
        
        forecaster = XGBoostForecaster()
        data = forecaster.load_data()
        
        if location not in forecaster.models:
            # Train model if not loaded
            forecaster.train_model(data, location)
        
        forecast_df = forecaster.generate_forecast(data, location, days)
        
        predictions = []
        for _, row in forecast_df.iterrows():
            predictions.append({
                "date": row['ds'].strftime('%Y-%m-%d') if hasattr(row['ds'], 'strftime') else str(row['ds']),
                "predicted_admissions": max(0, round(row['yhat'], 2)),
                "model": "xgboost"
            })
        return predictions
        
    except Exception as e:
        logger.error(f"XGBoost forecast error: {e}")
        # Return simple projection as fallback
        return generate_simple_forecast(location, days)


def generate_prophet_forecast(location: str, days: int) -> List[Dict]:
    """Generate forecast using Prophet model"""
    try:
        # Try to load saved forecast
        forecast_path = f"models/forecasts/prophet_{location.replace(' ', '_').lower()}_forecast.csv"
        if os.path.exists(forecast_path):
            forecast_df = pd.read_csv(forecast_path)
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            
            predictions = []
            for _, row in forecast_df.head(days).iterrows():
                predictions.append({
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "predicted_admissions": max(0, round(row['yhat'], 2)),
                    "lower_bound": max(0, round(row.get('yhat_lower', row['yhat'] * 0.8), 2)),
                    "upper_bound": max(0, round(row.get('yhat_upper', row['yhat'] * 1.2), 2)),
                    "model": "prophet"
                })
            return predictions
        
        # Generate simple forecast as fallback
        return generate_simple_forecast(location, days)
        
    except Exception as e:
        logger.error(f"Prophet forecast error: {e}")
        return generate_simple_forecast(location, days)


def generate_simple_forecast(location: str, days: int) -> List[Dict]:
    """Generate simple forecast based on historical averages"""
    df = load_forecast_data()
    
    if df.empty:
        # Return placeholder if no data
        predictions = []
        base_date = datetime.now()
        for i in range(days):
            predictions.append({
                "date": (base_date + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                "predicted_admissions": 5.0,  # Default value
                "model": "fallback"
            })
        return predictions
    
    location_data = df[df['location'] == location]
    if location_data.empty:
        avg = df['y'].mean()
    else:
        avg = location_data['y'].mean()
    
    last_date = df['ds'].max()
    predictions = []
    
    for i in range(days):
        pred_date = last_date + timedelta(days=i+1)
        # Add some variation based on day of week
        dow_factor = 1.0 if pred_date.weekday() < 5 else 0.85
        
        predictions.append({
            "date": pred_date.strftime('%Y-%m-%d'),
            "predicted_admissions": round(avg * dow_factor, 2),
            "model": "historical_average"
        })
    
    return predictions


# ============== API Endpoints ==============

@app.get("/", tags=["Root"])
async def root():
    """API Root - Welcome message"""
    return {
        "message": "Hospital Demand Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "locations": "/locations",
            "predict": "/predict",
            "metrics": "/metrics"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    data_available = os.path.exists(get_data_path())
    
    models_loaded = {
        "xgboost": os.path.exists("models/xgboost_metrics.json"),
        "prophet": os.path.exists("models/prophet_metrics.json") or os.path.exists("models/forecasts")
    }
    
    return HealthResponse(
        status="healthy" if data_available else "degraded",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_loaded,
        data_available=data_available
    )


@app.get("/locations", response_model=List[LocationInfo], tags=["Locations"])
async def get_locations():
    """Get list of available locations with statistics"""
    df = load_forecast_data()
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")
    
    locations = []
    for loc in df['location'].unique():
        loc_data = df[df['location'] == loc]
        locations.append(LocationInfo(
            name=loc,
            total_records=len(loc_data),
            date_range={
                "start": loc_data['ds'].min().strftime('%Y-%m-%d'),
                "end": loc_data['ds'].max().strftime('%Y-%m-%d')
            },
            avg_daily_admissions=round(loc_data['y'].mean(), 2)
        ))
    
    return locations


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """Generate demand forecast for a location"""
    available_locations = get_available_locations()
    
    if not available_locations:
        raise HTTPException(status_code=404, detail="No data available")
    
    if request.location not in available_locations:
        raise HTTPException(
            status_code=400, 
            detail=f"Location '{request.location}' not found. Available: {available_locations}"
        )
    
    # Generate predictions
    if request.model.lower() == "xgboost":
        predictions = generate_xgboost_forecast(request.location, request.days)
    elif request.model.lower() == "prophet":
        predictions = generate_prophet_forecast(request.location, request.days)
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{request.model}' not supported. Use 'xgboost' or 'prophet'"
        )
    
    return PredictionResponse(
        location=request.location,
        model=request.model,
        predictions=predictions,
        generated_at=datetime.now().isoformat()
    )


@app.get("/predict/{location}", tags=["Predictions"])
async def predict_get(
    location: str,
    days: int = Query(default=14, ge=1, le=90),
    model: str = Query(default="xgboost")
):
    """GET endpoint for predictions (convenience method)"""
    request = PredictionRequest(location=location, days=days, model=model)
    return await predict(request)


@app.get("/metrics", tags=["Metrics"])
async def get_all_metrics():
    """Get performance metrics for all models"""
    xgboost_metrics = load_metrics("xgboost")
    prophet_metrics = load_metrics("prophet")
    
    return {
        "xgboost": xgboost_metrics if xgboost_metrics else {"status": "no metrics available"},
        "prophet": prophet_metrics if prophet_metrics else {"status": "no metrics available"},
        "comparison": {
            "best_model": "xgboost" if xgboost_metrics else "prophet",
            "note": "XGBoost generally performs better with advanced features"
        }
    }


@app.get("/metrics/{model}", response_model=MetricsResponse, tags=["Metrics"])
async def get_model_metrics(model: str):
    """Get performance metrics for a specific model"""
    if model.lower() not in ["xgboost", "prophet"]:
        raise HTTPException(
            status_code=400, 
            detail="Model must be 'xgboost' or 'prophet'"
        )
    
    metrics = load_metrics(model.lower())
    
    if not metrics:
        raise HTTPException(
            status_code=404, 
            detail=f"No metrics found for model '{model}'"
        )
    
    return MetricsResponse(model=model, metrics=metrics)


@app.get("/features", tags=["Info"])
async def get_feature_info():
    """Get information about features used in models"""
    importance_path = "models/xgboost_feature_importance.json"
    
    if os.path.exists(importance_path):
        with open(importance_path, 'r') as f:
            importance = json.load(f)
        
        # Get top 20 features for each location
        top_features = {}
        for loc, features in importance.items():
            sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:20]
            top_features[loc] = [{"name": k, "importance": round(v, 4)} for k, v in sorted_features]
        
        return {
            "total_features": 99,
            "categories": ["holiday", "weather", "demographic", "temporal", "lagged", "fourier"],
            "top_features_by_location": top_features
        }
    
    return {"message": "Feature importance not available. Run XGBoost training first."}


# ============== Startup/Shutdown Events ==============

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Hospital Forecasting API...")
    logger.info(f"Data path: {get_data_path()}")
    logger.info(f"Available locations: {get_available_locations()}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Hospital Forecasting API...")


# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
