#!/usr/bin/env python3
"""
Setup script for Hospital Forecasting Project
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required!")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "models/saved_models",
        "logs",
        "reports",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def init_database():
    """Initialize SQLite database"""
    db_path = "data/hospital_forecasting.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hospital_admissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE NOT NULL,
            location VARCHAR(100) NOT NULL,
            admission_count INTEGER NOT NULL,
            disease_type VARCHAR(50),
            age_group VARCHAR(20),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hospital_capacity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location VARCHAR(100) NOT NULL,
            total_beds INTEGER NOT NULL,
            icu_beds INTEGER NOT NULL,
            available_beds INTEGER NOT NULL,
            date DATE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location VARCHAR(100) NOT NULL,
            forecast_date DATE NOT NULL,
            predicted_admissions INTEGER NOT NULL,
            confidence_lower INTEGER,
            confidence_upper INTEGER,
            model_used VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {db_path}")

def main():
    """Main setup function"""
    print("🏥 Hospital Forecasting Project Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Initialize database
    init_database()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. python src/data_ingestion/collect_sample_data.py")
    print("2. jupyter notebook notebooks/01_data_exploration.ipynb")
    print("3. Follow IMPLEMENTATION_GUIDE.md")

if __name__ == "__main__":
    main()