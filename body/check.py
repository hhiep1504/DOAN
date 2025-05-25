import pandas as pd

# Đọc file CSV
df = pd.read_csv('data/body/pdestre/body_annotations/split_annotations/body_train.csv')

# Sửa đổi đường dẫn
df['image_path'] = df['image_path'].str.replace('\\', '/')

# Lưu lại file CSV
df.to_csv('data/body/pdestre/body_annotations/split_annotations/body_train.csv', index=False)


