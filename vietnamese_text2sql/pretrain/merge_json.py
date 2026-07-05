import json
from pathlib import Path

# Định nghĩa đường dẫn thư mục chứa file (sửa lại cho đúng cấu trúc máy bạn nếu cần)
# Ở đây mặc định các file nằm chung thư mục với file script này
DATA_DIR = Path("C:/Users/Admin/Documents/Uni/DS319/Text-to-SQL/data/spider_data") 

file_paths = [
    DATA_DIR / "train_spider.json",
    DATA_DIR / "train_others_spider.json",
    DATA_DIR / "dev_spider.json"
]

output_path = DATA_DIR / "total_spider.json"

merged_data = []

print("🔄 Bắt đầu gộp các file JSON...")

for path in file_paths:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Kiểm tra đảm bảo dữ liệu đọc ra đúng dạng List
            if isinstance(data, list):
                merged_data += data
                print(f"✅ Đã đọc {len(data):>5,} câu hỏi từ file: {path.name}")
            else:
                print(f"⚠️ Cảnh báo: File {path.name} không phải cấu trúc dạng List.")
    else:
        print(f"❌ Lỗi: Không tìm thấy file {path.name}")

# Ghi toàn bộ dữ liệu đã gộp ra file mới
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print("-" * 50)
print(f"🎉 GỘP THÀNH CÔNG! File tổng: {output_path.name}")
print(f"📊 Tổng số lượng mẫu dữ liệu sau khi gộp: {len(merged_data):,} câu hỏi.")