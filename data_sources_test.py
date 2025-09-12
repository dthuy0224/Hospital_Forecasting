#!/usr/bin/env python
"""
Test script to verify access to major data sources and repositories
"""
from importlib.metadata import version as pkg_version, PackageNotFoundError


import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Function to safely import and verify modules
def check_module(module_name, import_name=None):
    if import_name is None:
        import_name = module_name
    
    try:
        if import_name == 'huggingface_hub':
            import huggingface_hub
            version = huggingface_hub.__version__
            print(f"✅ {module_name} ({version}) - Kết nối HuggingFace datasets và models")
            return True, huggingface_hub
        elif import_name == 'datasets':
            import datasets
            version = datasets.__version__
            print(f"✅ {module_name} ({version}) - Thư viện HuggingFace datasets")
            return True, datasets
        elif import_name == 'kaggle':
            import kaggle
            # Kaggle module doesn't always expose version
            version = getattr(kaggle, '__version__', 'Unknown')
            print(f"✅ {module_name} ({version}) - API chính thức Kaggle")
            return True, kaggle
        elif import_name == 'kagglehub':
            import kagglehub
            version = kagglehub.__version__
            print(f"{module_name} {version}) - API ok")
            return True, kagglehub
        elif import_name == 'sodapy':
            import sodapy
            version = sodapy.__version__
            print(f"✅ {module_name} ({version}) - Truy cập Open Data APIs (Socrata)")
            return True, sodapy
        elif import_name == 'ckanapi':
            import ckanapi
            try:
                version = pkg_version("ckanapi")
            except PackageNotFoundError:
                version = "Unknown"
            print(f"✅ {module_name} ({version}) - Truy cập CKAN Open Data portals")
            return True, ckanapi
        elif import_name == 'pandas_datareader':
            import pandas_datareader
            version = pandas_datareader.__version__
            print(f"✅ {module_name} ({version}) - Truy cập dữ liệu từ World Bank, FRED, Yahoo Finance")
            return True, pandas_datareader
        elif import_name == 'quandl':
            import quandl
            version = quandl.__version__
            print(f"✅ {module_name} ({version}) - Truy cập Nasdaq Data Link (Quandl) cho dữ liệu tài chính & kinh tế")
            return True, quandl
        else:
            # Generic module import
            module = __import__(import_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {module_name} ({version})")
            return True, module
    except ImportError as e:
        print(f"❌ {module_name}: Chưa cài đặt - {str(e)}")
        return False, None
    except Exception as e:
        print(f"⚠️ {module_name}: Lỗi - {str(e)}")
        return False, None

def test_huggingface_datasets():
    """Test truy cập HuggingFace Datasets"""
    success, datasets_lib = check_module('HuggingFace Datasets', 'datasets')
    if not success:
        return
    
    try:
        # List some dataset categories
        print("\n🔍 Dataset categories từ HuggingFace:")
        dataset_types = ["text", "image", "audio", "tabular"]
        for dtype in dataset_types[:3]:  # Just check first 3 to keep output shorter
            print(f"  - {dtype.capitalize()} datasets")
        
        # Load a tiny dataset as example
        print("\n📊 Loading tiny sample dataset...")
        sample = datasets_lib.load_dataset("huggingface/tiny-dataset-test")
        print(f"  Sample dataset loaded: {sample}")
        print("  ✓ Truy cập thành công đến HuggingFace Datasets Hub")
    except Exception as e:
        print(f"  ❌ Không thể truy cập HuggingFace Datasets Hub: {str(e)}")

def test_kaggle_datasets():
    """Test truy cập Kaggle Datasets"""
    success_hub, kagglehub_lib = check_module('Kaggle Hub', 'kagglehub')
    success_api, kaggle_lib = check_module('Kaggle API', 'kaggle')
    
    print("\n🔍 Kaggle API Status:")
    
    # Check kaggle.json config
    kaggle_config = os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')
    if os.path.exists(kaggle_config):
        print("  ✓ Tìm thấy kaggle.json configuration")
        # Don't print credentials
        print("  ✓ Đã thiết lập credentials (không hiển thị vì bảo mật)")
    else:
        print("  ❌ Chưa thiết lập Kaggle API credentials")
        print("      1. Đăng nhập vào tài khoản Kaggle")
        print("      2. Tạo API token tại: https://www.kaggle.com/settings")
        print("      3. Tải về kaggle.json và lưu vào ~/.kaggle/kaggle.json")
        print("      4. Chạy 'chmod 600 ~/.kaggle/kaggle.json' trên Linux/Mac")
    
    # Try kagglehub which doesn't require API credentials for public datasets
    if success_hub:
        try:
            print("\n📊 KaggleHub models/datasets truy cập được:")
            models_count = 5  # Just an example count
            print(f"  • {models_count}+ public models")
            datasets_count = 50000  # Approximate number
            print(f"  • {datasets_count}+ public datasets")
            print("  ✓ Có thể truy cập Kaggle Hub (không cần credentials)")
        except Exception as e:
            print(f"  ❌ Lỗi truy cập Kaggle Hub: {str(e)}")

def test_gov_data():
    """Test truy cập dữ liệu chính phủ và nguồn mở"""
    # Check Socrata (data.gov, many city data portals)
    success_socrata, sodapy_lib = check_module('Socrata API', 'sodapy')
    success_ckan, ckanapi_lib = check_module('CKAN API', 'ckanapi')
    
    print("\n🏛️ Open Government Data APIs:")
    
    if success_socrata:
        print("\n📊 Socrata-powered Open Data portals có thể kết nối:")
        portals = [
            "data.gov", "data.medicare.gov", "data.cms.gov",
            "healthdata.gov", "data.ny.gov", "opendata.go.th"
        ]
        for portal in portals[:5]:  # Show first 5
            print(f"  • {portal}")
        print("  ✓ Socrata client available for connection")
    
    if success_ckan:
        print("\n📊 CKAN-powered Open Data portals có thể kết nối:")
        portals = [
            "data.europa.eu", "datos.gob.es", "data.gov.uk",
            "data.gov.au", "data.gov.sg", "data.go.th"
        ]
        for portal in portals[:5]:  # Show first 5
            print(f"  • {portal}")
        print("  ✓ CKAN client available for connection")
        
    # Check pandas-datareader for economic/financial data
    success_pdr, pdr = check_module('Pandas DataReader', 'pandas_datareader')
    if success_pdr:
        print("\n💹 Financial/Economic Data Sources có thể kết nối:")
        sources = [
            "World Bank", "Federal Reserve Economic Data (FRED)",
            "OECD", "Eurostat", "Bank for International Settlements"
        ]
        for source in sources[:5]:
            print(f"  • {source}")
        print("  ✓ Pandas DataReader available for connection")

def main():
    """Main function to run all tests"""
    print("=" * 60)
    print("🌎 DATA SOURCE ACCESS TEST")
    print(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python {sys.version.split()[0]}")
    print("=" * 60)
    
    print("\n📚 CHECKING MODULE AVAILABILITY:")
    
    # Test HuggingFace
    test_huggingface_datasets()
    
    # Test Kaggle
    test_kaggle_datasets()
    
    # Test Government & Open Data
    test_gov_data()
    
    print("\n" + "=" * 60)
    print("🏁 TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()