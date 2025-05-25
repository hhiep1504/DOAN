import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_accessories_distribution():
    # Đọc file CSV
    df = pd.read_csv('data/body/hiep_dataset/body_annotations/body_annotations_with_accessories.csv')
    
    # Phân bố các loại accessories
    accessories_counts = df['accessories'].value_counts()
    total_samples = len(df)
    
    print("\nPhân bố các loại accessories:")
    print("-" * 50)
    for label, count in accessories_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{label}: {count} samples ({percentage:.2f}%)")
    
    # Vẽ biểu đồ phân bố
    plt.figure(figsize=(10, 6))
    sns.barplot(x=accessories_counts.index, y=accessories_counts.values)
    plt.title('Phân bố các loại accessories')
    plt.xlabel('Loại accessories')
    plt.ylabel('Số lượng mẫu')
    plt.show()
    
    # Phân bố theo nhóm đề xuất
    accessories_mapping = {
        0: 'Bag',
        1: 'Backpack',
        2: 'Other',
        3: 'Nothing'
    }
    
    # Tạo cột mới với tên nhóm
    df['accessories_group'] = df['accessories'].map(accessories_mapping)
    group_counts = df['accessories_group'].value_counts()
    
    print("\nPhân bố theo nhóm đề xuất:")
    print("-" * 50)
    for group, count in group_counts.items():
        percentage = (count / total_samples) * 100
        print(f"{group}: {count} samples ({percentage:.2f}%)")
    
    # Vẽ biểu đồ phân bố theo nhóm
    plt.figure(figsize=(10, 6))
    sns.barplot(x=group_counts.index, y=group_counts.values)
    plt.title('Phân bố theo nhóm accessories')
    plt.xlabel('Nhóm accessories')
    plt.ylabel('Số lượng mẫu')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    analyze_accessories_distribution() 