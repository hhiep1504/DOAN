import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(image_dir, train_annotation, val_annotation, test_annotation, num_beard_classes=3, num_glasses_classes=4):
    """
    Phân tích dữ liệu dataset
    
    Args:
        image_dir (str): Đường dẫn đến thư mục chứa ảnh
        train_annotation (str): Đường dẫn đến file annotation training
        val_annotation (str): Đường dẫn đến file annotation validation
        test_annotation (str): Đường dẫn đến file annotation test
        num_beard_classes (int): Số lớp cho râu
        num_glasses_classes (int): Số lớp cho kính
    """
    print("\n=== Phân tích dữ liệu ===")
    
    # Định nghĩa tên các lớp
    beard_classes = {
        0: 'Có râu',
        1: 'Không râu',
        2: 'Không rõ'
    }
    
    glasses_classes = {
        0: 'Kính thường',
        1: 'Kính râm',
        2: 'Không kính',
        3: 'Không rõ'
    }
    
    # Phân tích từng tập dữ liệu
    for name, annotation_file in [
        ('Training', train_annotation),
        ('Validation', val_annotation),
        ('Test', test_annotation)
    ]:
        print(f"\n=== Tập {name} ===")
        df = pd.read_csv(annotation_file)
        print(f"Tổng số mẫu: {len(df)}")
        
        # Phân tích nhãn râu
        print("\nPhân bố nhãn râu:")
        beard_counts = df['beard'].value_counts().sort_index()
        for label, count in beard_counts.items():
            print(f"  {beard_classes.get(label, f'Lớp {label}')}: {count} mẫu")
        
        # Phân tích nhãn kính
        print("\nPhân bố nhãn kính:")
        glasses_counts = df['glasses'].value_counts().sort_index()
        for label, count in glasses_counts.items():
            print(f"  {glasses_classes.get(label, f'Lớp {label}')}: {count} mẫu")
        
        # Kiểm tra nhãn không hợp lệ
        invalid_beard = df[~df['beard'].between(0, num_beard_classes-1)]
        invalid_glasses = df[~df['glasses'].between(0, num_glasses_classes-1)]
        
        if len(invalid_beard) > 0:
            print("\nNhãn râu không hợp lệ:")
            print(invalid_beard[['imagejpg', 'beard']])
            
        if len(invalid_glasses) > 0:
            print("\nNhãn kính không hợp lệ:")
            print(invalid_glasses[['imagejpg', 'glasses']])
        
        # Kiểm tra ảnh tồn tại
        missing_images = []
        for img_path in df['imagejpg']:
            full_path = os.path.join(image_dir, img_path)
            if not os.path.exists(full_path):
                missing_images.append(img_path)
        
        if missing_images:
            print(f"\nẢnh không tồn tại trong tập {name}:")
            print(missing_images)
        else:
            print(f"\nTất cả ảnh trong tập {name} đều tồn tại.")
        
        # Vẽ biểu đồ phân bố
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Biểu đồ phân bố râu
        sns.barplot(x=beard_counts.index, y=beard_counts.values, ax=ax1)
        ax1.set_title(f'Phân bố nhãn râu - Tập {name}')
        ax1.set_xlabel('Nhãn râu')
        ax1.set_ylabel('Số lượng')
        ax1.set_xticklabels([beard_classes.get(i, f'Lớp {i}') for i in beard_counts.index])
        
        # Biểu đồ phân bố kính
        sns.barplot(x=glasses_counts.index, y=glasses_counts.values, ax=ax2)
        ax2.set_title(f'Phân bố nhãn kính - Tập {name}')
        ax2.set_xlabel('Nhãn kính')
        ax2.set_ylabel('Số lượng')
        ax2.set_xticklabels([glasses_classes.get(i, f'Lớp {i}') for i in glasses_counts.index])
        
        plt.tight_layout()
        plt.savefig(f'data_analysis_{name.lower()}.png')
        plt.close()
    
    print("\n=== Kết thúc phân tích ===")

if __name__ == '__main__':
    # Cấu hình đường dẫn
    config = {
        'image_dir': 'data/head/pdestre/head_images',
        'train_annotation': 'data/head/pdestre/head_annotations/head_train.csv',
        'val_annotation': 'data/head/pdestre/head_annotations/head_val.csv',
        'test_annotation': 'data/head/pdestre/head_annotations/head_test.csv',
        'num_beard_classes': 3,
        'num_glasses_classes': 4
    }
    
    # Phân tích dữ liệu
    analyze_dataset(**config) 