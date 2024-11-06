from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV với mã hóa utf-8
file_path = r'C:\python\binhluan.csv'  # Thay đổi đường dẫn tới file CSV của bạn
df = pd.read_csv(file_path, encoding='utf-8')

# Kiểm tra xem có dữ liệu trong cột 'Comment'
if 'Comment' not in df.columns:
    print("Cột 'Comment' không tồn tại trong dữ liệu.")
else:
    # Chọn cột 'Comment' để tạo Word Cloud
    text = " ".join(comment for comment in df['Comment'])

    # Tạo Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap='Blues', contour_color='black').generate(text)

    # Vẽ biểu đồ Word Cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Không hiển thị trục
    plt.title('Word Cloud từ các bình luận', fontsize=16)
    plt.show()
