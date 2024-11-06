import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Đọc dữ liệu từ file CSV
file_path = r'C:\python\binhluan.csv'
df = pd.read_csv(file_path)

# Tính toán TF-IDF cho các sentiment
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Sentiment'])

# Tạo DataFrame từ TF-IDF
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Tính trọng số TF-IDF cho từng hàng
tfidf_values = tfidf_df.values

# Tạo tọa độ cho biểu đồ
x_coords = []
y_coords = []
colors = []

# Lặp qua từng hàng để tạo tọa độ cho biểu đồ
for index, sentiment in enumerate(df['Sentiment']):
    # Sử dụng giá trị TF-IDF của hàng tương ứng
    for feature_index, value in enumerate(tfidf_values[index]):
        if value > 0:  # Chỉ vẽ cho các giá trị lớn hơn 0
            # Thêm ngẫu nhiên vào tọa độ để tạo sự lẫn lộn
            x_offset = np.random.uniform(-0.2, 0.2)  # Đẩy tọa độ x một chút
            y_offset = np.random.uniform(-0.1, 0.1)  # Đẩy tọa độ y một chút
            
            x_coords.append(feature_index + x_offset)  # Thay đổi x với offset
            y_coords.append(value + y_offset)  # Thay đổi y với offset
            colors.append('blue' if sentiment == 'positive' else 'red')  # Màu cho sentiment

# Vẽ biểu đồ phân tán
plt.figure(figsize=(12, 8))

# Vẽ dấu chấm cho từng giá trị
plt.scatter(x_coords, y_coords, alpha=0.7, c=colors, s=30, edgecolor='w', linewidth=0.5)

# Thêm nhãn cho các trục
plt.title('Tọa độ phân bố TF-IDF cho Sentiment', fontsize=16)
plt.xlabel('Các tính năng', fontsize=14)
plt.ylabel('Trọng số TF-IDF', fontsize=14)

# Thay đổi tick cho trục x để dễ nhìn
plt.xticks(ticks=np.arange(len(vectorizer.get_feature_names_out())), labels=vectorizer.get_feature_names_out(), rotation=90)

# Tinh chỉnh hiển thị
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()