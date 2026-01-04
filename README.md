# 🏥 Hospital Demand Forecasting

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Accuracy%2095.85%25-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

> AI-powered hospital admission forecasting system with **95.85% accuracy**

## 📊 Features

- **Real Data**: Kaggle HDHI Hospital dataset (15,757 records)
- **Advanced ML**: XGBoost + Prophet with Optuna hyperparameter tuning
- **99 Features**: Holidays, weather, demographics, Fourier seasonality
- **REST API**: FastAPI with Swagger docs
- **Docker Ready**: One-command deployment

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Kaggle    │────▶│  Preprocess │────▶│   Models    │────▶│   API/UI    │
│    Data     │     │  99 Features│     │ XGB+Prophet │     │   FastAPI   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
docker-compose up -d
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

### Option 2: Local
```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py

# Run API
python -m uvicorn src.api.main:app --reload
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/locations` | List locations |
| GET | `/predict/{location}` | Get forecast |
| GET | `/metrics` | Model performance |
| GET | `/docs` | Swagger UI |

### Example Request
```bash
curl "http://localhost:8000/predict/Ludhiana%20Central?days=7&model=xgboost"
```

## 📈 Model Performance

| Model | Accuracy | MAPE |
|-------|----------|------|
| XGBoost | **95.85%** | 4.15% |
| Prophet | 35.60% | 64.40% |

## 🗂️ Project Structure

```
├── src/
│   ├── api/              # FastAPI REST API
│   ├── data_ingestion/   # Kaggle data collector
│   ├── data_processing/  # Feature engineering
│   ├── models/           # Prophet, XGBoost
│   └── visualization/    # Streamlit dashboard
├── config/               # Configuration
├── data/                 # Data files
├── models/               # Saved models
├── Dockerfile
└── docker-compose.yml
```

## ⚙️ Requirements

- Python 3.11+
- Kaggle API credentials (`~/.kaggle/kaggle.json`)
- Docker (optional)

## 📄 License

MIT License

---