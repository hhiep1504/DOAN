import pandas as pd
import os

def fix_invalid_labels(train_file, val_file, test_file, num_beard_classes=3, num_glasses_classes=4):
    """
    Sửa các nhãn không hợp lệ trong file CSV
    
    Args:
        train_file (str): Đường dẫn đến file train CSV
        val_file (str): Đường dẫn đến file validation CSV
        test_file (str): Đường dẫn đến file test CSV
        num_beard_classes (int): Số lớp cho râu
        num_glasses_classes (int): Số lớp cho kính
    """
    print("\n=== Sửa nhãn không hợp lệ ===")
    
    # Xử lý từng file
    for name, file_path in [
        ('Training', train_file),
        ('Validation', val_file),
        ('Test', test_file)
    ]:
        print(f"\nXử lý file {name}...")
        
        # Đọc file CSV
        df = pd.read_csv(file_path)
        original_count = len(df)
        
        # Tìm và sửa nhãn râu không hợp lệ
        invalid_beard = df[~df['beard'].between(0, num_beard_classes-1)]
        if len(invalid_beard) > 0:
            print(f"\nTìm thấy {len(invalid_beard)} nhãn râu không hợp lệ:")
            print(invalid_beard[['imagejpg', 'beard']])
            
            # Sửa nhãn không hợp lệ về 0 (Có râu)
            df.loc[~df['beard'].between(0, num_beard_classes-1), 'beard'] = 0
            print("Đã sửa các nhãn râu không hợp lệ về 0 (Có râu)")
        
        # Tìm và sửa nhãn kính không hợp lệ
        invalid_glasses = df[~df['glasses'].between(0, num_glasses_classes-1)]
        if len(invalid_glasses) > 0:
            print(f"\nTìm thấy {len(invalid_glasses)} nhãn kính không hợp lệ:")
            print(invalid_glasses[['imagejpg', 'glasses']])
            
            # Sửa nhãn không hợp lệ về 0 (Kính thường)
            df.loc[~df['glasses'].between(0, num_glasses_classes-1), 'glasses'] = 0
            print("Đã sửa các nhãn kính không hợp lệ về 0 (Kính thường)")
        
        # Lưu file đã sửa
        backup_path = file_path.replace('.csv', '_backup.csv')
        os.rename(file_path, backup_path)
        df.to_csv(file_path, index=False)
        
        print(f"\nĐã lưu bản sao lưu tại: {backup_path}")
        print(f"Đã lưu file đã sửa tại: {file_path}")
        
        # In thống kê
        print(f"\nThống kê sau khi sửa:")
        print(f"Tổng số mẫu: {len(df)}")
        print("\nPhân bố nhãn râu:")
        print(df['beard'].value_counts().sort_index())
        print("\nPhân bố nhãn kính:")
        print(df['glasses'].value_counts().sort_index())
    
    print("\n=== Hoàn thành sửa nhãn ===")

if __name__ == '__main__':
    # Cấu hình đường dẫn
    config = {
        'train_file': 'data/head/pdestre/head_annotations/head_train.csv',
        'val_file': 'data/head/pdestre/head_annotations/head_val.csv',
        'test_file': 'data/head/pdestre/head_annotations/head_test.csv',
        'num_beard_classes': 3,
        'num_glasses_classes': 4
    }
    
    # Sửa nhãn không hợp lệ
    fix_invalid_labels(**config) 