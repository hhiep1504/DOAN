import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class HeadDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Args:
            image_dir (str): Đường dẫn đến thư mục chứa ảnh
            annotation_file (str): Đường dẫn đến file annotation
            transform (callable, optional): Transform áp dụng cho ảnh
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # Đọc file annotation
        self.df = pd.read_csv(annotation_file)
        
        # Đảm bảo các cột cần thiết tồn tại
        required_columns = ['imagejpg', 'beard', 'glasses']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Chuyển đổi nhãn thành số nguyên
        self.df['beard'] = self.df['beard'].astype(int)
        self.df['glasses'] = self.df['glasses'].astype(int)
        
        # Định nghĩa các class
        self.beard_classes = {
            0: 'Có râu',
            1: 'Không râu',
            2: 'Không rõ'
        }
        
        self.glasses_classes = {
            0: 'Kính thường',
            1: 'Kính râm',
            2: 'Không kính',
            3: 'Không rõ'
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            # Lấy đường dẫn ảnh và nhãn
            img_path = os.path.join(self.image_dir, self.df.iloc[idx]['imagejpg'])
            beard_label = self.df.iloc[idx]['beard']
            glasses_label = self.df.iloc[idx]['glasses']
            
            # Đọc ảnh
            image = Image.open(img_path).convert('RGB')
            
            # Áp dụng transform nếu có
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'beard': beard_label,
                'glasses': glasses_label
            }
            
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None

def create_data_loaders(image_dir, annotation_file, batch_size=32, train_ratio=0.8, val_ratio=0.1):
    """
    Tạo data loaders cho training, validation và testing
    
    Args:
        image_dir (str): Đường dẫn đến thư mục chứa ảnh
        annotation_file (str): Đường dẫn đến file annotation
        batch_size (int): Kích thước batch
        train_ratio (float): Tỷ lệ dữ liệu training
        val_ratio (float): Tỷ lệ dữ liệu validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Đọc file annotation
    df = pd.read_csv(annotation_file)
    
    # Đảm bảo các cột cần thiết tồn tại
    required_columns = ['imagejpg', 'beard', 'glasses']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Kiểm tra số lượng mẫu cho mỗi lớp
    beard_counts = df['beard'].value_counts()
    glasses_counts = df['glasses'].value_counts()
    
    print("\nSố lượng mẫu cho mỗi lớp:")
    print("\nRâu:")
    for label, count in beard_counts.items():
        print(f"  Lớp {label}: {count} mẫu")
    print("\nKính:")
    for label, count in glasses_counts.items():
        print(f"  Lớp {label}: {count} mẫu")
    
    # Chia dữ liệu thành train, val, test
    try:
        # Thử chia với stratify
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio,
            random_state=42,
            stratify=df[['beard', 'glasses']].values
        )
        
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            random_state=42,
            stratify=temp_df[['beard', 'glasses']].values
        )
    except ValueError as e:
        print(f"\nKhông thể chia dữ liệu với stratify: {e}")
        print("Chuyển sang chia dữ liệu ngẫu nhiên...")
        
        # Chia ngẫu nhiên nếu không thể stratify
        train_df, temp_df = train_test_split(
            df, 
            train_size=train_ratio,
            random_state=42
        )
        
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            random_state=42
        )
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs('data/annotations', exist_ok=True)
    
    # Lưu các file annotation
    train_df.to_csv('data/annotations/head_train.csv', index=False)
    val_df.to_csv('data/annotations/head_val.csv', index=False)
    test_df.to_csv('data/annotations/head_test.csv', index=False)
    
    print("\nSố lượng mẫu sau khi chia:")
    print(f"Train: {len(train_df)} mẫu")
    print(f"Val: {len(val_df)} mẫu")
    print(f"Test: {len(test_df)} mẫu")
    
    # Định nghĩa transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Tạo datasets
    train_dataset = HeadDataset(
        image_dir=image_dir,
        annotation_file='data/annotations/head_train.csv',
        transform=train_transform
    )
    
    val_dataset = HeadDataset(
        image_dir=image_dir,
        annotation_file='data/annotations/head_val.csv',
        transform=val_transform
    )
    
    test_dataset = HeadDataset(
        image_dir=image_dir,
        annotation_file='data/annotations/head_test.csv',
        transform=val_transform
    )
    
    # Tạo data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """
    Custom collate function để xử lý các batch có thể chứa None
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    # Gộp các dictionary trong batch
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'beard': torch.tensor([item['beard'] for item in batch]),
        'glasses': torch.tensor([item['glasses'] for item in batch])
    }

def visualize_samples(dataset, num_samples=4, figsize=(12, 8)):
    """
    Visualize samples from dataset
    
    Args:
        dataset: Dataset to visualize
        num_samples: Number of samples to show
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        img = sample['image']
        
        # Convert tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        # Get labels
        beard_label = sample['beard']
        glasses_label = sample['glasses']
        
        beard_text = dataset.beard_classes.get(beard_label, f"Unknown ({beard_label})")
        glasses_text = dataset.glasses_classes.get(glasses_label, f"Unknown ({glasses_label})")
        
        axes[i].set_title(f"Râu: {beard_text}\nKính: {glasses_text}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Test dataset loading
    print("Testing dataset loading...")
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            image_dir='data/head/pdestre/head_images',
            annotation_file='data/head/pdestre/head_annotations/head_annotations.csv',
            batch_size=16
        )
        
        # Test một batch
        for batch in train_loader:
            if batch is not None:
                print("Batch shape:", batch['image'].shape)
                print("Beard labels:", batch['beard'].shape)
                print("Glasses labels:", batch['glasses'].shape)
                break
                
        print("Dataset loading successful!")
        
    except Exception as e:
        print(f"Error during dataset loading: {e}")