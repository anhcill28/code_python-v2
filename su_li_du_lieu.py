import re
import pandas as pd
import sys

# Thiết lập mã hóa cho stdout
sys.stdout.reconfigure(encoding='utf-8')

# Đường dẫn đến file gốc và file kết quả
input_file_path = r'C:\python\full_train_1.csv'
output_file_path = r'C:\python\binhluan.csv'

# Đọc file CSV
try:
    data = pd.read_csv(input_file_path)
    print("File đã được tải thành công.")
except FileNotFoundError:
    print("Lỗi: Không tìm thấy file tại đường dẫn đã cung cấp.")
    exit()

# Kiểm tra các cột chính
required_columns = ['Comment', 'Rating']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    print(f"Lỗi: File thiếu các cột cần thiết: {', '.join(missing_columns)}.")
    exit()

# Hàm xử lý văn bản bình luận
def preprocess_comment(comment):
    comment = re.sub(r'[^\w\s]', '', comment).lower()
    replacements = {
        "ko": "không",
        "k": "không",
        "vs": "với",
        "15k": "15000",
        "ship": "vận chuyển",
    "shop": "cửa hàng",
    "m": "mình",
    "mik": "mình",
    "ko": "không",
    "k": " không ",
    "kh": "không",
    "khong": "không",
    "kg": "không",
    "khg": "không",
    "tl": "trả lời",
    "r": "rồi",
    "fb": "mạng xã hội", # facebook
    "face": "mạng xã hội",
    "thanks": "cảm ơn",
    "thank": "cảm ơn",
    "tks": "cảm ơn",
    "tk": "cảm ơn",
    "ok": "tốt",
    "dc": "được",
    "vs": "với",
    "đt": "điện thoại",
    "thjk": "thích",
    "qá": "quá",
    "trể": "trễ",
    "bgjo": "bao giờ",
    "good": "tốt",
    "bh": "bây giờ",
    "sale": "giảm giá",
    "ntn": "như thế này",
    "vote": "đánh giá tốt",
    "ms": "mới",
    "hnay": "hôm nay",
    "kute": "dễ thương",
    "bik": "biết",
    "od": "gọi món",
    "mn": "mọi người",
    "c": "chị",
    "đc": "được",
    "uk": "ừ",
    "t": "tôi",
    "tt": "thứ tự",
    "gj": "gì",
    "j": "gì",
    "đx": "được",
    "m": "mày",
    "zậy": "vậy",
    "wa": "qua",
    "zui": "vui",
    "thik": "thích",
    "add": "thêm",
    "pko": "phải không",
    "cmt": "bình luận",
    "dt": "dễ thương",
    "ib": "inbox",
    "klq": "không liên quan",
    "nx": "nhận xét",
    "rep": "trả lời",
    "dj": "đi",
    "mog": "mong",
    "bít": "biết",
    "nc": "nước",
    "lun": "luôn",
    "hiu": "hiểu",
    "rui": "rồi",
    "thui": "thôi",
    "view": "phong cảnh",
    "đg": "đang",
    "h": "giờ",
    "zòn": "giòn",
    "cx": "cũng",
    "kbiet": "không biết",
    "đ": "đéo",
    "mk": "mình",
    "trc": "trước",
    "bùn": "buồn",
    "iu": "yêu",
    "vs": "với",
    "lua": "lừa",
    "b": "bạn",
    }
    for word, replacement in replacements.items():
        comment = re.sub(r'\b' + re.escape(word) + r'\b', replacement, comment)
    return comment

data['Comment'] = data['Comment'].astype(str).apply(preprocess_comment)
initial_row_count = len(data)
data = data.dropna(subset=['Comment', 'Rating'])
data = data[data['Comment'].str.strip() != ""]
rows_dropped = initial_row_count - len(data)
print(f"Số hàng bị xóa do thiếu dữ liệu: {rows_dropped}")

def assign_sentiment(rating):
    try:
        rating = float(rating)
       return 'positive' if rating == 1 else 'negative'
    except ValueError:
        print(f"Lỗi: Giá trị không phải là số: {rating}")
        return None

data['Sentiment'] = data['Rating'].apply(assign_sentiment)
data = data.dropna(subset=['Sentiment'])

# Lưu dữ liệu đã xử lý vào một file CSV mới
try:
    data[['Comment', 'Rating', 'Sentiment']].to_csv(output_file_path, index=False, encoding="utf-8")
    print(f"Dữ liệu đã được lưu thành công vào file: {output_file_path}")
except Exception as e:
    print(f"Lỗi khi lưu file: {e}")
