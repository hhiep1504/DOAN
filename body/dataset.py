import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

class BodyDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Args:
            image_dir (str): Đường dẫn đến thư mục gốc chứa ảnh body (ví dụ: data/body/pdestre/body_images)
            annotation_file (str): Đường dẫn đến file annotation body CSV (ví dụ: data/body/pdestre/body_images/body_annotations_with_accessories.csv)
            transform (callable, optional): Transform áp dụng cho ảnh
        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Đọc file annotation
        self.df = pd.read_csv(annotation_file)

        # Đảm bảo cột cần thiết tồn tại (chỉ image_path và accessories)
        required_columns = ['image_path', 'accessories']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Chuyển đổi nhãn accessories thành số nguyên
        self.df['accessories'] = self.df['accessories'].astype(int)

        # Định nghĩa các class cho Body (chỉ accessories)
        self.accessories_classes = {
            0: 'Bag',
            1: 'Backpack',
            2: 'Other',
            3: 'Nothing'
        }
        # Bỏ các dictionary cho gender và age

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            # Lấy đường dẫn ảnh và nhãn accessories
            # image_path trong CSV là tương đối so với self.image_dir
            img_relative_path = self.df.iloc[idx]['image_path']
            img_full_path = self.image_dir / img_relative_path

            # Đọc ảnh
            image = Image.open(img_full_path).convert('RGB')

            # Áp dụng transform nếu có
            if self.transform:
                image = self.transform(image)

            # Trả về ảnh và nhãn accessories
            item = {
                'image': image,
                'accessories': self.df.iloc[idx]['accessories']
            }

            # Bỏ các nhãn gender và age

            return item

        except FileNotFoundError:
             print(f"Error loading image: File not found at {img_full_path}")
             return None
        except Exception as e:
            print(f"Error loading data at index {idx} from {img_full_path}: {e}")
            return None

# Sử dụng collate_fn để xử lý None (từ __getitem__)
def body_collate_fn(batch):
    """
    Custom collate function cho body dataset (chỉ với accessories)
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Gộp các dictionary trong batch
    # Chỉ cần image và accessories
    collated_batch = {
        'image': torch.stack([item['image'] for item in batch]),
        'accessories': torch.tensor([item['accessories'] for item in batch], dtype=torch.long)
    }

    return collated_batch

def create_body_data_loaders(image_dir, annotation_file, batch_size=32, train_ratio=0.8, val_ratio=0.1):
    """
    Tạo data loaders cho training, validation và testing body dataset

    Args:
        image_dir (str): Đường dẫn đến thư mục gốc chứa ảnh body
        annotation_file (str): Đường dẫn đến file annotation body CSV
        batch_size (int): Kích thước batch
        train_ratio (float): Tỷ lệ dữ liệu training
        val_ratio (float): Tỷ lệ dữ liệu validation

    Returns:
        train_loader, val_loader, test_loader
    """
    # Đọc file annotation
    df = pd.read_csv(annotation_file)
    
    # Kiểm tra và lọc các ảnh không tồn tại
    print("\nKiểm tra sự tồn tại của các file ảnh...")
    image_dir = Path(image_dir)
    valid_rows = []
    missing_images = []
    
    for idx, row in df.iterrows():
        img_path = image_dir / row['image_path']
        if img_path.exists():
            valid_rows.append(row)
        else:
            missing_images.append(row['image_path'])
    
    if missing_images:
        print(f"\nTìm thấy {len(missing_images)} ảnh không tồn tại:")
        for img in missing_images[:5]:  # In ra 5 ảnh đầu tiên để kiểm tra
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... và {len(missing_images) - 5} ảnh khác")
    
    # Cập nhật DataFrame với chỉ các dòng hợp lệ
    df = pd.DataFrame(valid_rows)
    print(f"\nSố lượng mẫu sau khi lọc: {len(df)}")

    # Đảm bảo các cột cần thiết tồn tại (chỉ image_path và accessories)
    required_columns = ['image_path', 'accessories']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in annotation file: {col}")

    # Kiểm tra số lượng mẫu cho lớp accessories
    accessories_counts = df['accessories'].value_counts()
    print("\nSố lượng mẫu cho mỗi lớp accessories:")
    for label, count in accessories_counts.items():
        print(f"  Lớp {label}: {count} mẫu")

    # Các cột để stratify. Chỉ sử dụng accessories
    stratify_cols = ['accessories']

    # Chia dữ liệu thành train, val, test
    try:
        # Thử chia với stratify
        print("\nĐang chia dữ liệu với stratify...")
        train_df, temp_df = train_test_split(
            df,
            train_size=train_ratio,
            random_state=42,
            stratify=df[stratify_cols].values # Stratify dựa trên accessories
        )

        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio_adjusted,
            random_state=42,
            stratify=temp_df[stratify_cols].values # Stratify temp_df dựa trên accessories
        )
        print("Chia dữ liệu với stratify thành công!")

    except ValueError as e:
        print(f"\nKhông thể chia dữ liệu với stratify ({stratify_cols}): {e}")
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
        print("Chia dữ liệu ngẫu nhiên thành công!")

    # Tạo thư mục output cho các file annotation chia nhỏ
    output_annotation_dir = Path(annotation_file).parent / "split_annotations" # Lưu trong thư mục con 'split_annotations' cùng nơi file annotation gốc
    output_annotation_dir.mkdir(parents=True, exist_ok=True)

    # Lưu các file annotation chia nhỏ
    train_annotation_path = output_annotation_dir / "body_train.csv"
    val_annotation_path = output_annotation_dir / "body_val.csv"
    test_annotation_path = output_annotation_dir / "body_test.csv"

    train_df.to_csv(train_annotation_path, index=False)
    val_df.to_csv(val_annotation_path, index=False)
    test_df.to_csv(test_annotation_path, index=False)


    print("\nSố lượng mẫu sau khi chia:")
    print(f"Train: {len(train_df)} mẫu (lưu tại: {train_annotation_path})")
    print(f"Val: {len(val_df)} mẫu (lưu tại: {val_annotation_path})")
    print(f"Test: {len(test_df)} mẫu (lưu tại: {test_annotation_path})")

    # Định nghĩa transforms (có thể cần điều chỉnh cho ảnh body)
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(), # Ảnh body có thể không cần flip ngang nếu dáng người quan trọng
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)), # Kích thước có thể cần khác
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Mean/Std có thể khác cho ảnh body
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Kích thước có thể cần khác
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Mean/Std có thể khác cho ảnh body
                           std=[0.229, 0.224, 0.225])
    ])

    # Tạo datasets
    train_dataset = BodyDataset(
        image_dir=image_dir,
        annotation_file=train_annotation_path, # Dùng file annotation đã chia nhỏ
        transform=train_transform
    )

    val_dataset = BodyDataset(
        image_dir=image_dir,
        annotation_file=val_annotation_path, # Dùng file annotation đã chia nhỏ
        transform=val_transform
    )

    test_dataset = BodyDataset(
        image_dir=image_dir,
        annotation_file=test_annotation_path, # Dùng file annotation đã chia nhỏ
        transform=val_transform
    )

    # Tạo data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, # Có thể cần điều chỉnh num_workers
        pin_memory=True,
        collate_fn=body_collate_fn # Sử dụng custom collate fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4, # Có thể cần điều chỉnh num_workers
        pin_memory=True,
        collate_fn=body_collate_fn # Sử dụng custom collate fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4, # Có thể cần điều chỉnh num_workers
        pin_memory=True,
        collate_fn=body_collate_fn # Sử dụng custom collate fn
    )

    return train_loader, val_loader, test_loader

# Hàm visualize cũng cần cập nhật nếu muốn hiển thị nhãn accessories
def visualize_body_samples(dataset, num_samples=4, figsize=(12, 8)):
    """
    Visualize samples from body dataset (chỉ hiển thị nhãn accessories)

    Args:
        dataset: BodyDataset to visualize
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
            mean = np.array([0.485, 0.456, 0.406]) # Sử dụng mean/std tương ứng với transform
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)

        axes[i].imshow(img)

        # Get accessories label
        accessories_label = sample['accessories']
        accessories_text = dataset.accessories_classes.get(accessories_label, f"Unknown ({accessories_label})")

        axes[i].set_title(f"Accessories: {accessories_text}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Ví dụ sử dụng cho dataset P-DESTRE body
    # Cần đảm bảo file body_annotations_with_accessories.csv đã được tạo
    # PDESTRE_BODY_IMAGE_DIR = 'data/body/pdestre/body_images' # Thư mục chứa ảnh body P-DESTRE
    # PDESTRE_BODY_ANNOTATION_FILE = 'data/body/pdestre/body_annotations/body_annotations_with_accessories.csv' # File annotation cuối cùng

    # print("Testing body dataset loading for P-DESTRE...")
    # try:
    #     train_loader, val_loader, test_loader = create_body_data_loaders(
    #         image_dir=PDESTRE_BODY_IMAGE_DIR,
    #         annotation_file=PDESTRE_BODY_ANNOTATION_FILE,
    #         batch_size=16
    #     )

    #     # Test một batch
    #     for i, batch in enumerate(train_loader):
    #         if batch is not None:
    #             print(f"\nSample batch {i+1}:")
    #             print("Batch image shape:", batch['image'].shape)
    #             if 'accessories' in batch:
    #                 print("Accessories labels shape:", batch['accessories'].shape)
    #                 print("Accessories labels sample:", batch['accessories'][:5]) # Print vài sample
    #             # Bỏ kiểm tra gender và age
    #             break # Chỉ test 1 batch

    #     print("\nBody dataset loading successful!")

    #     # Ví dụ visualize
    #     print("\nVisualizing sample images...")
    #     visualize_body_samples(train_loader.dataset, num_samples=4)


    # except Exception as e:
    #     print(f"\nError during body dataset loading: {e}")


    # Ví dụ sử dụng cho dataset Hiep body
    # Cần đảm bảo file body_annotations_with_accessories.csv đã được tạo cho Hiep
    HIEP_BODY_IMAGE_DIR = 'data/body/hiep_dataset/body_images' # Thư mục chứa ảnh body Hiep
    HIEP_BODY_ANNOTATION_FILE = 'data/body/hiep_dataset/body_annotations/body_annotations_with_accessories.csv' # File annotation cuối cùng

    print("\nTesting body dataset loading for Hiep...")
    try:
        train_loader, val_loader, test_loader = create_body_data_loaders(
            image_dir=HIEP_BODY_IMAGE_DIR,
            annotation_file=HIEP_BODY_ANNOTATION_FILE,
            batch_size=16
        )

        # Test một batch
        for i, batch in enumerate(train_loader):
            if batch is not None:
                print(f"\nSample batch {i+1}:")
                print("Batch image shape:", batch['image'].shape)
                if 'accessories' in batch:
                    print("Accessories labels shape:", batch['accessories'].shape)
                    print("Accessories labels sample:", batch['accessories'][:5])
                # Bỏ kiểm tra gender và age
                break # Chỉ test 1 batch

        print("\nBody dataset loading successful!")

    except Exception as e:
        print(f"\nError during body dataset loading for Hiep: {e}") 