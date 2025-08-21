#!/usr/bin/env python3
"""
Complete Pipeline Runner for Hospital Forecasting Project
Runs the entire data science pipeline from setup to dashboard
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Complete pipeline orchestrator"""
    
    def __init__(self):
        self.steps = [
            ("🔧 Setup", self.run_setup),
            ("📊 Data Collection", self.run_data_collection),
            ("🧹 Data Processing", self.run_data_processing),
            ("🤖 Model Training", self.run_model_training),
            ("📈 Dashboard Launch", self.launch_dashboard)
        ]
        
    def run_step(self, step_name: str, func):
        """Run a pipeline step with error handling"""
        logger.info(f"Starting: {step_name}")
        start_time = time.time()
        
        try:
            func()
            duration = time.time() - start_time
            logger.info(f"✅ Completed: {step_name} ({duration:.1f}s)")
            return True
        except Exception as e:
            logger.error(f"❌ Failed: {step_name} - {str(e)}")
            return False
    
    def run_setup(self):
        """Run setup script"""
        subprocess.check_call([sys.executable, "scripts/setup.py"])
    
    def run_data_collection(self):
        """Run data collection"""
        subprocess.check_call([sys.executable, "src/data_ingestion/collect_sample_data.py"])
    
    def run_data_processing(self):
        """Run data processing"""
        subprocess.check_call([sys.executable, "src/data_processing/preprocess_data.py"])
    
    def run_model_training(self):
        """Run model training"""
        subprocess.check_call([sys.executable, "src/models/prophet_forecasting.py"])
    
    def launch_dashboard(self):
        """Launch Streamlit dashboard"""
        logger.info("🚀 Launching dashboard... (This will open in your browser)")
        logger.info("Press Ctrl+C to stop the dashboard")
        
        # Run Streamlit in the background
        cmd = [sys.executable, "-m", "streamlit", "run", "src/visualization/streamlit_dashboard.py", "--server.headless", "false"]
        subprocess.call(cmd)
    
    def run_pipeline(self, start_from: int = 0):
        """Run the complete pipeline"""
        logger.info("🏥 Hospital Forecasting Pipeline Started")
        logger.info("=" * 50)
        
        total_steps = len(self.steps)
        completed_steps = 0
        
        for i, (step_name, func) in enumerate(self.steps[start_from:], start_from):
            if self.run_step(step_name, func):
                completed_steps += 1
            else:
                logger.error(f"Pipeline failed at step {i+1}: {step_name}")
                logger.info(f"To resume from this step, run: python run_pipeline.py --start-from {i}")
                return False
        
        logger.info(f"\n🎉 Pipeline completed successfully!")
        logger.info(f"✅ {completed_steps}/{total_steps} steps completed")
        logger.info(f"\n📱 Access your dashboard at: http://localhost:8501")
        
        return True
    
    def create_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            "models/forecasts",
            "logs",
            "reports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

def print_usage():
    """Print usage instructions"""
    print("""
🏥 Hospital Forecasting Pipeline Runner

Usage:
    python run_pipeline.py                 # Run complete pipeline
    python run_pipeline.py --start-from 2  # Start from step 2
    python run_pipeline.py --help          # Show this help

Pipeline Steps:
    0. 🔧 Setup
    1. 📊 Data Collection  
    2. 🧹 Data Processing
    3. 🤖 Model Training
    4. 📈 Dashboard Launch

Requirements:
    - Python 3.8+
    - All dependencies in requirements.txt
    - Sufficient disk space (>100MB)

Output:
    - Trained models in models/ directory
    - Processed data in data/processed/
    - Interactive dashboard at http://localhost:8501
    """)

def main():
    """Main function"""
    # Parse command line arguments
    start_from = 0
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print_usage()
            return
        elif sys.argv[1] == '--start-from' and len(sys.argv) > 2:
            try:
                start_from = int(sys.argv[2])
            except ValueError:
                logger.error("Invalid start-from value. Must be an integer.")
                return
    
    # Initialize pipeline
    runner = PipelineRunner()
    runner.create_directories()
    
    # Run pipeline
    success = runner.run_pipeline(start_from)
    
    if success:
        print(f"\n🎉 SUCCESS! Your Hospital Forecasting system is ready!")
        print(f"📱 Dashboard: http://localhost:8501")
        print(f"📊 Data: data/processed/")
        print(f"🤖 Models: models/")
        print(f"📋 Logs: pipeline.log")
    else:
        print(f"\n❌ Pipeline failed. Check pipeline.log for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()