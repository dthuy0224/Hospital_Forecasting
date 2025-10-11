# Git Setup Guide
## Hospital Demand Forecasting Project

### 📋 Before Committing to GitHub

#### 1. **Environment Variables & Secrets**
Tạo file `.env` (không commit) với các thông tin nhạy cảm:
```bash
# Database credentials
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=hospital_forecasting

# API Keys
OPENAI_API_KEY=your_openai_key
WEATHER_API_KEY=your_weather_key

# Other secrets
SECRET_KEY=your_secret_key
```

#### 2. **Git Configuration**
```bash
# Set your git config
git config user.name "Your Name"
git config user.email "your.email@example.com"

# Initialize repository
git init
git add .
git commit -m "Initial commit: Hospital Forecasting Project"

# Add remote repository
git remote add origin https://github.com/yourusername/hospital-forecasting.git
git branch -M main
git push -u origin main
```

#### 3. **Protected Files (.gitignore)**
Các file sau đã được bảo vệ bởi `.gitignore`:
- ✅ `venv/` - Virtual environment
- ✅ `__pycache__/` - Python cache files
- ✅ `*.pyc` - Compiled Python files
- ✅ `data/raw/` - Raw data files
- ✅ `data/processed/` - Processed data files
- ✅ `*.db`, `*.sqlite` - Database files
- ✅ `models/saved_models/` - Trained model files
- ✅ `logs/` - Log files
- ✅ `*.log` - Log files
- ✅ `.env` - Environment variables
- ✅ API keys và credentials
- ✅ Temporary files
- ✅ IDE files (.vscode/, .idea/)
- ✅ OS files (.DS_Store, Thumbs.db)

#### 4. **Repository Structure**
```
hospital-forecasting/
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── run_pipeline.py           # Main pipeline script
├── config/
│   └── config.yaml           # Configuration file
├── src/                      # Source code
│   ├── data_ingestion/
│   ├── data_processing/
│   ├── models/
│   └── visualization/
├── tests/                    # Test files
├── notebooks/                # Jupyter notebooks
├── reports/                  # Analysis reports
└── docs/                     # Documentation
```

#### 5. **First Commit Checklist**
- [ ] ✅ Created `.gitignore`
- [ ] ✅ Cleaned up project structure
- [ ] ✅ Removed sensitive files
- [ ] ✅ Removed cache files
- [ ] ✅ Removed temporary files
- [ ] ✅ Removed old optimization files
- [ ] ✅ Created proper directory structure
- [ ] ✅ Added `.gitkeep` files for empty directories

#### 6. **Security Best Practices**
1. **Never commit:**
   - API keys
   - Database passwords
   - Personal information
   - Large data files
   - Model weights (if too large)

2. **Use environment variables:**
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   api_key = os.getenv('API_KEY')
   ```

3. **Use .env files:**
   ```bash
   # .env (not committed)
   API_KEY=your_secret_key
   DB_PASSWORD=your_password
   ```

#### 7. **Git Commands**
```bash
# Check status
git status

# Add files
git add .

# Commit changes
git commit -m "Descriptive commit message"

# Push to remote
git push origin main

# Pull latest changes
git pull origin main
```

#### 8. **Branch Strategy**
```bash
# Create feature branch
git checkout -b feature/new-feature

# Work on feature
git add .
git commit -m "Add new feature"

# Switch back to main
git checkout main

# Merge feature
git merge feature/new-feature
```

### 🚀 Ready for GitHub!

Project đã được cleanup và sẵn sàng để commit lên GitHub. Tất cả các file nhạy cảm và không cần thiết đã được loại bỏ hoặc bảo vệ bởi `.gitignore`.

**Next Steps:**
1. Tạo repository trên GitHub
2. Add remote origin
3. Push code lên GitHub
4. Tiếp tục với Performance Optimization task
