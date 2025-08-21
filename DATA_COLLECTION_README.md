# Data Collection Guide - Hospital Forecasting Project

## 📊 Script: `src/data_ingestion/collect_sample_data.py`

Script này tạo dữ liệu mẫu cho hệ thống dự báo nhu cầu bệnh viện, bao gồm:
- **Admission data**: Số ca nhập viện theo ngày, địa phương, loại bệnh, nhóm tuổi
- **Capacity data**: Số giường bệnh, ICU, giường khả dụng theo địa phương

## 🚀 Cách sử dụng

### 1. Chạy cơ bản (khuyến nghị)
```bash
python src/data_ingestion/collect_sample_data.py
```

### 2. Tùy chỉnh khoảng thời gian
```bash
# Tạo dữ liệu 1 năm (2024)
python src/data_ingestion/collect_sample_data.py --start-date 2024-01-01 --end-date 2024-12-31

# Tạo dữ liệu 6 tháng gần đây
python src/data_ingestion/collect_sample_data.py --start-date 2024-07-01 --end-date 2024-12-31
```

### 3. Bao gồm dữ liệu COVID-19 từ API
```bash
python src/data_ingestion/collect_sample_data.py --include-covid-api
```

### 4. Sử dụng file config tùy chỉnh
```bash
python src/data_ingestion/collect_sample_data.py --config my_config.yaml
```

## 📁 Output Files

Script sẽ tạo các file sau:

### Database
- `data/hospital_forecasting.db` - SQLite database với 2 bảng:
  - `hospital_admissions` - Dữ liệu nhập viện
  - `hospital_capacity` - Dữ liệu giường bệnh

### CSV Files
- `data/raw/hospital_admissions.csv` - Dữ liệu nhập viện
- `data/raw/hospital_capacity.csv` - Dữ liệu giường bệnh  
- `data/raw/combined_hospital_data.csv` - Dữ liệu kết hợp

### Reports
- `reports/data_collection_summary.json` - Báo cáo tổng hợp

## 🏥 Dữ liệu được tạo

### Địa phương (10 tỉnh/thành)
- Ho Chi Minh City, Ha Noi, Da Nang, Can Tho, Hai Phong
- Nha Trang, Hue, Vung Tau, Bien Hoa, Thu Dau Mot

### Loại bệnh (7 loại)
- COVID-19, Flu, Pneumonia, Heart Disease
- Diabetes, Dengue Fever, Others

### Nhóm tuổi (5 nhóm)
- 0-18, 19-30, 31-50, 51-65, 65+

### Đặc điểm dữ liệu
- **Seasonal patterns**: Mùa cúm (12-3), mùa sốt xuất huyết (6-10)
- **Weekly patterns**: Ít nhập viện vào cuối tuần
- **Holiday effects**: Giảm nhập viện vào ngày lễ
- **Age-specific**: Phân bố khác nhau theo bệnh và tuổi
- **Location-based**: Tỷ lệ nhập viện theo dân số địa phương

## 🧪 Test

Chạy test để kiểm tra script hoạt động:

```bash
python test_data_collection.py
```

Test sẽ kiểm tra:
- ✅ Tạo dữ liệu thành công
- ✅ Schema đúng định dạng
- ✅ Lưu vào database và CSV
- ✅ Tích hợp với pipeline hiện có

## 📊 Ví dụ Output

### Data Summary
```json
{
  "generated_at": "2024-01-15T10:30:00",
  "date_range": {
    "start": "2023-01-01",
    "end": "2024-12-31", 
    "days": 731
  },
  "locations": {
    "count": 10,
    "list": ["Bien Hoa", "Can Tho", "Da Nang", ...]
  },
  "records": {
    "admissions": 255500,
    "capacity": 7310,
    "total": 262810
  },
  "statistics": {
    "avg_daily_admissions": 349.5,
    "avg_total_beds": 1500.0
  }
}
```

## 🔧 Tùy chỉnh

### Thêm địa phương mới
Chỉnh sửa `config/config.yaml`:
```yaml
locations:
  - "Ho Chi Minh City"
  - "Ha Noi"
  - "Your New Location"  # Thêm vào đây
```

### Thay đổi tỷ lệ nhập viện
Chỉnh sửa `base_rates` trong `generate_admission_data()`:
```python
base_rates = {
    "Ho Chi Minh City": 150,  # Thay đổi số này
    "Ha Noi": 120,
    # ...
}
```

### Thêm loại bệnh mới
Chỉnh sửa `disease_types` trong `__init__()`:
```python
self.disease_types = [
    "COVID-19", "Flu", "Your New Disease"  # Thêm vào đây
]
```

## ⚠️ Lưu ý

1. **Dữ liệu mẫu**: Đây là dữ liệu giả lập, không phải dữ liệu thật
2. **Performance**: Tạo 2 năm dữ liệu có thể mất 1-2 phút
3. **Storage**: Cần ~50MB cho 2 năm dữ liệu
4. **API**: COVID-19 API có thể không khả dụng, script sẽ bỏ qua

## 🔄 Pipeline Integration

Sau khi chạy data collection, tiếp tục với:

```bash
# 1. Xử lý dữ liệu
python src/data_processing/preprocess_data.py

# 2. Huấn luyện mô hình
python src/models/prophet_forecasting.py

# 3. Chạy dashboard
streamlit run src/visualization/streamlit_dashboard.py

# Hoặc chạy toàn bộ pipeline
python run_pipeline.py
```

## 🐛 Troubleshooting

### Lỗi Import
```bash
# Cài đặt dependencies
pip install -r requirements.txt
```

### Lỗi Database
```bash
# Xóa và tạo lại database
# Windows:
Remove-Item -Path .\data\hospital_forecasting.db -Force
# Linux/Mac:
rm data/hospital_forecasting.db

# Chạy lại
python src/data_ingestion/collect_sample_data.py
```

### Lỗi API COVID-19
- Script sẽ tự động bỏ qua nếu API không khả dụng
- Không ảnh hưởng đến dữ liệu mẫu khác

---

**🎉 Chúc mừng! Bạn đã hoàn thành bước thu thập dữ liệu cho dự án Hospital Forecasting!** 