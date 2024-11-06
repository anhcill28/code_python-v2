from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV với mã hóa utf-8
file_path = r'C:\python\binhluan.csv'
df = pd.read_csv(file_path, encoding='utf-8')

# Kiểm tra dữ liệu nếu cần (bỏ qua nếu không cần thiết)
# print(df.head())

# Chuyển đổi cảm xúc thành số (0: Tiêu cực, 1: Tích cực, 2: Trung lập)
sentiment_mapping = {'negative': 0, 'positive': 1, 'neutral': 2}
df['sentiment_numeric'] = df['Sentiment'].map(sentiment_mapping)

# Khởi tạo TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Tạo một DataFrame để lưu các giá trị TF-IDF cho từng cảm xúc
tfidf_df = pd.DataFrame()

# Lặp qua các cảm xúc và tính toán TF-IDF
for sentiment in df['Sentiment'].unique():
    # Lọc các bình luận theo cảm xúc
    sentiment_data = df[df['Sentiment'] == sentiment]['Comment']
    
    # Tính toán TF-IDF cho mỗi cảm xúc
    X_tfidf = tfidf_vectorizer.fit_transform(sentiment_data)
    
    # Tạo DataFrame với các từ và giá trị TF-IDF tương ứng
    tfidf_values = X_tfidf.mean(axis=0).A1  # Tính giá trị trung bình của các từ
    tfidf_df_sentiment = pd.DataFrame(list(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_values)),
                                      columns=['Word', f'TF-IDF_{sentiment}'])
    
    # Ghép vào DataFrame chính
    tfidf_df = pd.merge(tfidf_df, tfidf_df_sentiment, how='outer', on='Word') if not tfidf_df.empty else tfidf_df_sentiment

# Vẽ biểu đồ các từ có TF-IDF cao nhất cho mỗi cảm xúc
top_words = 10
plt.figure(figsize=(12, 8))

for i, sentiment in enumerate(df['Sentiment'].unique()):
    sentiment_col = f'TF-IDF_{sentiment}'
    sentiment_data = tfidf_df[['Word', sentiment_col]].sort_values(by=sentiment_col, ascending=False).head(top_words)
    
    # Vẽ biểu đồ
    plt.subplot(2, 2, i + 1)
    plt.barh(sentiment_data['Word'], sentiment_data[sentiment_col], color='skyblue')
    plt.title(f'Top {top_words} từ với TF-IDF cao nhất cho cảm xúc {sentiment}', fontsize=14)
    plt.xlabel('TF-IDF', fontsize=12)
    plt.ylabel('Từ', fontsize=12)

plt.tight_layout()
plt.show()
