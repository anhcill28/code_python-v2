import sys
import pandas as pd
import matplotlib.pyplot as plt

# Cài đặt lại mã hóa đầu ra của terminal thành 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

# Đọc dữ liệu từ file CSV với encoding='utf-8'
file_path = r'C:\python\binhluan.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Kiểm tra dữ liệu
print(df.head())

# Đếm số lượng các giá trị trong cột Sentiment
sentiment_counts = df['Sentiment'].value_counts()

# Vẽ biểu đồ phân bổ cảm xúc
plt.figure(figsize=(10, 6))
sentiment_counts.plot(kind='bar', color=['#1f77b4', '#ff7f0e'], edgecolor='black')

# Thêm chú thích cho mỗi thanh
for i, v in enumerate(sentiment_counts):
    plt.text(i, v + 5, str(v), ha='center', va='bottom', fontsize=12, color='black')

# Cải thiện tiêu đề và nhãn trục
plt.title('Phân bố cảm xúc (Sentiment)', fontsize=16)
plt.xlabel('Cảm xúc', fontsize=14)
plt.ylabel('Số lượng', fontsize=14)

# Định dạng trục x
plt.xticks(rotation=0, fontsize=12)

# Định dạng trục y
plt.yticks(fontsize=12)

# Thêm lưới để dễ đọc hơn
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Cải thiện bố cục
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()

# Lưu kết quả với encoding='utf-8'
result_file_path = 'sentiment_distribution.png'
plt.savefig(result_file_path, dpi=300)

# In ra thông báo đã lưu kết quả
print(f'DataFrame phân bổ cảm xúc đã được lưu vào {result_file_path}')
