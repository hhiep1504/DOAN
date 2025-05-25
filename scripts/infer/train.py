import torch
import torch.nn as nn
import torch.optim as optim
# Thêm default_collate
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from torch.utils.data.dataloader import default_collate
from model import FeatureClassifier
from torchvision import transforms
import os
from pathlib import Path
from tqdm import tqdm
import math
import warnings
from PIL import Image
from dataset_local import PdestreFeatureDataset 

# Import CustomDataset và collate_skip_none
from dataset_custom import CustomDataset, collate_skip_none, FEATURES, CLASS_NAMES

# Định nghĩa danh sách features (phải khớp với thứ tự trong tensor features)
FEATURES = ['gender', 'age', 'ethnicity', 'beard', 'glasses', 'accessories']

# Định nghĩa các class names (cần giữ nguyên như trong compare.py hoặc nơi khác)
GENDER_NAMES = {0: 'Nam', 1: 'Nữ', 2: 'Unknown'}
AGE_NAMES = {
    0: '0-17', 1: '18-55', 2: '55+', 3: 'Unknown'
}
ETHNICITY_NAMES = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Unknown'}
BEARD_NAMES = {0: 'Có râu', 1: 'Không râu', 2: 'Không rõ'}
GLASSES_NAMES = {0: 'Kính thường', 1: 'Kính râm', 2: 'Không kính', 3: 'Không rõ'}
ACCESSORIES_NAMES = {
    0: 'Túi xách', 1: 'Ba lô', 2: 'Túi kéo', 3: 'Ô/Dù', 
    4: 'Túi thể thao', 5: 'Túi đi chợ', 6: 'Không có', 7: 'Không rõ' # -1: 'Error' không có trong file annotation, nên bỏ qua
}

# Danh sách các dictionary class names tương ứng với FEATURES
CLASS_NAMES = [GENDER_NAMES, AGE_NAMES, ETHNICITY_NAMES, BEARD_NAMES, GLASSES_NAMES, ACCESSORIES_NAMES]

# ======================================================================
# Configuration
# ======================================================================
# Đường dẫn cho P-DESTRE
PDESTRE_JPG_DIR = Path("data/processed/images")
PDESTRE_ANNOTATION_DIR = Path("data/processed/labels")

# Đường dẫn cho Custom Dataset (Hiep Dataset)
HIEP_BASE_DIR = Path("outputs/hiep_dataset") # Điều chỉnh đường dẫn này cho phù hợp với cấu trúc của bạn
HIEP_ANNOTATION_DIR = Path("data/processed/hiep_dataset") # Điều chỉnh đường dẫn này cho phù hợp

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 0.001
NUM_EPOCHS = 30
CHECKPOINT_DIR = Path("checkpoints")
BEST_MODEL_FILENAME = "best_weight_combined.pth" # Đổi tên file weight để phân biệt
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = CHECKPOINT_DIR / BEST_MODEL_FILENAME
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0001

USE_LR_SCHEDULER = True
LR_PATIENCE = 3
LR_FACTOR = 0.1
LR_MIN = 1e-6

# ======================================================================
# Helper Wrapper Dataset (Giữ nguyên) - Chỉ cần đảm bảo nó hoạt động với cả hai loại dataset
# ======================================================================
class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        # Attempt to get the original dataset reference for potential use later
        if hasattr(subset, 'dataset'):
             # Check if it's a Subset wrapping a ConcatDataset
             if isinstance(subset.dataset, ConcatDataset):
                 # Assume all concatenated datasets have the same properties (like norm_mean/std)
                 # Get the reference from the first dataset in the ConcatDataset
                 if subset.dataset.datasets:
                     # Lấy reference từ một trong các dataset gốc (Pdestre hoặc Custom)
                     # Đảm bảo rằng cả hai dataset đều có cấu trúc trả về tương tự nhau sau khi load
                     self.original_dataset = subset.dataset.datasets[0]
                 else:
                     self.original_dataset = None
             else: # It's a Subset wrapping a single dataset
                 self.original_dataset = subset.dataset
        else: # Not a standard Subset object, maybe the dataset itself
             self.original_dataset = None

    def __getitem__(self, index):
        try:
            # getItem từ subset có thể trả về tuple 2 hoặc 4 giá trị tùy dataset gốc và collate_fn
            # Nếu collate_fn của dataset gốc trả về None, subset[index] sẽ là None
            original_item = self.subset[index]
            
            if original_item is None:
                return None

            # Giả định dataset gốc (PdestreFeatureDataset và CustomDataset sau sửa đổi)
            # trả về (img, features) hoặc (img, features, original_img, sample_info)
            # Chúng ta chỉ cần img và features ở đây
            if isinstance(original_item, tuple) and len(original_item) >= 2:
                img, features = original_item[:2] # Lấy 2 phần tử đầu tiên
            else:
                 # unexpected item format
                 warnings.warn(f"Unexpected item format from subset[index] at index {index}. Type: {type(original_item)}")
                 return None


            # Apply transform only if image is valid (PIL Image)
            # Nếu dataset gốc đã áp dụng ToTensor, img có thể là tensor.
            # TransformedSubset này nên áp dụng transform cho cả PIL và Tensor nếu cần
            # Tuy nhiên, transform ở đây thường chỉ là augmentation trước ToTensor
            # Ta giả định transform ở đây là các bước trước ToTensor và ToTensor cuối cùng
            # Nếu dataset gốc trả về PIL, ta apply transform. Nếu trả về Tensor, ta giả định transform đã áp dụng hoặc không cần thêm.
            if isinstance(img, Image.Image): # Nếu là ảnh PIL, áp dụng transform đầy đủ
                 if self.transform:
                      img = self.transform(img)
                 # Nếu img là PIL nhưng transform là None, ta cần chuyển nó thành tensor
                 # Điều này phụ thuộc vào việc model mong đợi Tensor hay PIL
                 # Model thường mong đợi Tensor, nên cần đảm bảo ToTensor được áp dụng
                 # Cách tốt nhất là ToTensor là bước cuối cùng trong self.transform
            elif isinstance(img, torch.Tensor):
                 # Nếu đã là tensor, giả định transform đã áp dụng hoặc không cần thêm
                 pass # Không làm gì thêm
            else:
                 # unexpected type
                 warnings.warn(f"Item image at index {index} is not PIL Image or Tensor. Type: {type(img)}")
                 return None # Bỏ qua nếu không phải ảnh hợp lệ

            return img, features # Trả về ảnh (sau transform nếu là PIL, hoặc nguyên gốc nếu Tensor) và features

        except Exception as e:
             warnings.warn(f"Error in TransformedSubset __getitem__ for index {index}: {e}")
             return None

    # Thêm phương thức __len__
    def __len__(self):
        return len(self.subset)


# ======================================================================
# Custom Collate Function (Giữ nguyên) - Đảm bảo nó hoạt động với batch của (img, features)
# ======================================================================
def collate_skip_none(batch):
    # Loại bỏ các mẫu None
    batch = [item for item in batch if item is not None]
    if not batch:
        return None # Trả về None nếu batch rỗng sau khi lọc
    
    # Kiểm tra xem các item có đúng format (ảnh, features) không
    # Nếu TransformedSubset trả về (img, features, original_img, sample_info), cần điều chỉnh collate
    # Hiện tại TransformedSubset chỉ trả về (img, features)
    try:
        # default_collate mong đợi batch là list of tuples/lists of same structure
        # Ví dụ: [(img1, features1), (img2, features2)]
        # Nó sẽ trả về (collated_imgs, collated_features)
        return default_collate(batch)
    except Exception as e:
        warnings.warn(f"Error during default_collate: {e}. Skipping batch.")
        # Bạn có thể muốn kiểm tra nội dung batch ở đây nếu lỗi lặp lại
        # print("Problematic batch content (before collate):", batch)
        return None


# ======================================================================
# Validation Function (Sửa lại để xử lý features dict)
# ======================================================================
def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0
    # Correct predictions cho từng feature
    correct_preds = {feature: 0 for feature in FEATURES}
    total_samples = 0
    batches_processed = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validating', leave=False)
        for batch_data in progress_bar:
            if batch_data is None:
                warnings.warn("Skipping an empty batch in validation.")
                continue
            batches_processed += 1
            
            # batch_data từ DataLoader với collate_fn mặc định là tuple (images, features_tensor)
            # features_tensor là tensor 2D: (batch_size, num_features)
            images, features_tensor = batch_data # features_tensor (batch_size, 6)

            images = images.to(device)
            
            # Chuyển features_tensor thành dictionary of tensors cho từng feature
            # FEATURES = ['gender', 'age', 'ethnicity', 'beard', 'glasses', 'accessories']
            # features_tensor[:, 0] -> gender, features_tensor[:, 1] -> age, ...
            targets = {feature: features_tensor[:, i].to(device) for i, feature in enumerate(FEATURES)}

            try:
                outputs = model(images) # outputs là dictionary {'gender': tensor, 'age': tensor, ...}
                batch_loss = 0
                batch_size_actual = images.size(0) # Use actual batch size

                # Tính loss và accuracy cho từng feature
                valid_batch_for_loss = True
                for feature_name in FEATURES:
                    if feature_name in outputs and feature_name in targets:
                         # Check if criterion exists for this feature (should always be the case based on definition)
                         if feature_name in criterion:
                             loss = criterion[feature_name](outputs[feature_name], targets[feature_name])
                             batch_loss += loss.item() # Accumulate item loss only
                             _, predicted = outputs[feature_name].max(1)
                             correct_preds[feature_name] += predicted.eq(targets[feature_name]).sum().item()
                         else:
                             warnings.warn(f"No criterion defined for feature '{feature_name}'. Skipping loss/acc for this feature.")
                             valid_batch_for_loss = False # Consider batch invalid for overall loss if criterion missing
                    else:
                         warnings.warn(f"Missing feature '{feature_name}' in model outputs or targets during validation. Skipping loss/acc for this key.")
                         valid_batch_for_loss = False # Consider batch invalid

                # Accumulate total loss scaled by actual batch size
                # Chỉ cộng loss nếu batch được coi là hợp lệ cho tất cả features
                if valid_batch_for_loss:
                    total_val_loss += batch_loss * batch_size_actual # batch_loss is sum of avg losses per feature

                total_samples += batch_size_actual # Luôn đếm số mẫu đã xử lý

                # Calculate average loss for this specific batch for display
                # batch_loss ở đây là tổng loss trung bình trên batch cho tất cả features
                batch_avg_loss = batch_loss # criterion already gives avg loss for the batch
                progress_bar.set_postfix({'val_batch_loss': f"{batch_avg_loss:.4f}"})

            except Exception as e:
                warnings.warn(f"Error during validation batch {progress_bar.n}: {e}. Skipping batch.")
                # Clean up potentially problematic tensors
                del images, features_tensor, targets
                if 'outputs' in locals(): del outputs
                if 'loss' in locals(): del loss
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue # Skip to next batch on error

    # Calculate final average loss and accuracies over all valid samples processed
    # total_samples là tổng số mẫu đã qua vòng lặp, không hẳn là số mẫu có loss hợp lệ
    # Cần đếm số mẫu có loss hợp lệ nếu muốn tính avg loss chính xác hơn
    # Tuy nhiên, dùng total_samples ở đây là phổ biến
    avg_val_loss = total_val_loss / total_samples if total_samples > 0 else 0
    accuracies = {feature: (100. * correct_preds[feature] / total_samples) if total_samples > 0 else 0 for feature in FEATURES}
    print(f"\nValidation finished. Processed {total_samples} samples in {batches_processed} non-empty batches.")
    return avg_val_loss, accuracies


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0.0001, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait before stopping when loss is not improving
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
            verbose (bool): If True, prints a message for each improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        """
        Call after every validation to determine if training should stop.
        
        Args:
            val_loss (float): Validation loss from the current epoch
            
        Returns:
            bool: True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"  EarlyStopping: New best validation loss: {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} epochs without improvement (best: {self.best_loss:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  EarlyStopping: Stopping early at epoch due to no improvement after {self.patience} epochs")
        
        return self.early_stop


# ======================================================================
# Main Training Function
# ======================================================================
def train():
    # --- Thiết lập Device và Seed ---
    if RANDOM_SEED:
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU.")

    # --- Định nghĩa Transforms ---
    # Define transforms here - they will be applied via TransformedSubset
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

    # --- Tải các Datasets Riêng Lẻ ---
    print("\n--- Loading Individual Datasets ---")

    # Tải PdestreFeatureDataset
    pdestre_datasets = []
    if not PDESTRE_JPG_DIR.exists() or not PDESTRE_ANNOTATION_DIR.exists():
        print(f"Warning: P-DESTRE image directory '{PDESTRE_JPG_DIR}' or annotation directory '{PDESTRE_ANNOTATION_DIR}' does not exist. Skipping P-DESTRE dataset.")
    else:
        all_pdestre_annotation_files = sorted([f for f in PDESTRE_ANNOTATION_DIR.glob('*.txt') if not f.name.startswith('.')])
        if not all_pdestre_annotation_files:
            print(f"Warning: No .txt files found in {PDESTRE_ANNOTATION_DIR}. Skipping P-DESTRE dataset.")
        else:
            print(f"Found {len(all_pdestre_annotation_files)} P-DESTRE annotation files.")
            for ann_file in tqdm(all_pdestre_annotation_files, desc="Loading P-DESTRE datasets"):
                try:
                    # *** Cần đảm bảo PdestreFeatureDataset trả về nhãn tuổi 4 lớp ***
                    # Bạn cần sửa đổi file PdestreFeatureDataset.py để ánh xạ nhãn tuổi 8->4
                    # hoặc tạo một wrapper dataset cho Pdestre ở đây để làm việc đó.
                    dataset_instance = PdestreFeatureDataset( # Thay thế bằng lớp dataset P-DESTRE thực tế của bạn
                        jpg_dir=PDESTRE_JPG_DIR,
                        annotation_file=ann_file,
                        transform=None # Transform sẽ được áp dụng sau khi combine
                    )
                    if len(dataset_instance) > 0:
                        pdestre_datasets.append(dataset_instance)
                except Exception as e:
                     warnings.warn(f"\nCould not process P-DESTRE file {ann_file.name}. Error: {e}")

            print(f"\nSuccessfully loaded {len(pdestre_datasets)} P-DESTRE dataset(s). Total samples: {sum(len(d) for d in pdestre_datasets)}.")


    # Tải CustomDataset (Hiep Dataset)
    custom_datasets = []
    if not HIEP_BASE_DIR.exists() or not HIEP_ANNOTATION_DIR.exists():
         print(f"Warning: Custom dataset base directory '{HIEP_BASE_DIR}' or annotation directory '{HIEP_ANNOTATION_DIR}' does not exist. Skipping Custom dataset.")
    else:
        # CustomDataset._load_annotations đã tìm kiếm file .txt
        # Đảm bảo file custom_dataset.py chứa lớp CustomDataset và các định nghĩa cần thiết (FEATURES, CLASS_NAMES)
        try:
            # Pass transform=None here, it will be applied by TransformedSubset later
            custom_full_dataset_instance = CustomDataset(
                hiep_base_dir=HIEP_BASE_DIR,
                annotation_dir=HIEP_ANNOTATION_DIR,
                transform=None # Transform sẽ được áp dụng sau khi combine
            )
            if len(custom_full_dataset_instance) > 0:
                # CustomDataset đã load tất cả từ nhiều file .txt
                custom_datasets.append(custom_full_dataset_instance)
                print(f"Successfully loaded Custom dataset. Total samples: {len(custom_full_dataset_instance)}.")
            else:
                print("Custom dataset instance is empty after loading.")

        except Exception as e:
             warnings.warn(f"\nCould not load Custom Dataset. Error: {e}")


    # --- Kết hợp các Datasets ---
    print("\n--- Combining Datasets ---")
    all_datasets_list = pdestre_datasets + custom_datasets

    if not all_datasets_list:
        print("\nError: No datasets loaded to combine. Cannot proceed with training.")
        return

    combined_dataset = ConcatDataset(all_datasets_list)
    total_samples_full = len(combined_dataset)
    print(f"Total combined dataset size (before filtering Nones): {total_samples_full} samples.")


    # --- Chia toàn bộ Dataset thành Train/Val ---
    print("\n--- Splitting Full Dataset into Train/Validation ---")
    if total_samples_full == 0:
        print("Error: Combined dataset is empty. Cannot split.")
        return

    # Adjust ratios if they don't sum to 1
    if abs(TRAIN_RATIO + VAL_RATIO - 1.0) > 1e-6 :
        print("Warning: Train + Validation ratios do not sum to 1.0. Normalizing.")
        norm_factor = TRAIN_RATIO + VAL_RATIO
        train_ratio_adj = TRAIN_RATIO / norm_factor
        val_ratio_adj = VAL_RATIO / norm_factor
    else:
        train_ratio_adj = TRAIN_RATIO
        val_ratio_adj = VAL_RATIO

    train_size = int(train_ratio_adj * total_samples_full)
    val_size = total_samples_full - train_size

    # Ensure train and val sizes are positive if possible
    if total_samples_full > 0:
        if train_size == 0 and total_samples_full > 0:
            train_size = 1
            val_size = total_samples_full - 1
            print("Adjusted split: Train size 0, forcing Train=1, Val=total-1.")
        if val_size == 0 and total_samples_full > train_size:
             val_size = 1
             train_size = total_samples_full - 1
             print("Adjusted split: Val size 0, forcing Val=1, Train=total-1.")

    print(f"Splitting dataset into: Train={train_size}, Validation={val_size}")
    if train_size <= 0 or val_size <= 0:
        print("Error: Training or Validation set size is non-positive after split. Cannot train.")
        return

    try:
        # random_split cần list các kích thước nguyên
        train_subset, val_subset = random_split(combined_dataset, [train_size, val_size])
    except Exception as e:
            print(f"Error during random_split: {e}")
            return

    # --- Áp dụng Transforms bằng Wrapper ---
    print("\n--- Applying Transforms via Wrappers ---")
    train_dataset_transformed = TransformedSubset(train_subset, train_transform)
    val_dataset_transformed = TransformedSubset(val_subset, val_transform)
    print(f"Train dataset wrapper size: {len(train_dataset_transformed)}")
    print(f"Validation dataset wrapper size: {len(val_dataset_transformed)}")


    # --- Tạo DataLoaders ---
    print("\n--- Creating DataLoaders ---")
    train_loader = DataLoader(
        train_dataset_transformed,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_skip_none # Use custom collate
    )
    val_loader = DataLoader(
        val_dataset_transformed,
        batch_size=BATCH_SIZE, # Có thể dùng batch_size nhỏ hơn cho val
        shuffle=False,
        num_workers=NUM_WORKERS, # Có thể dùng num_workers nhỏ hơn cho val
        pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_skip_none # Use custom collate
    )
    # Note: len(loader) might be slightly inaccurate now due to skipped batches
    print(f"Train loader created. Batches (approx): {len(train_loader)}")
    print(f"Validation loader created. Batches (approx): {len(val_loader)}")


    # --- Tạo Model, Loss, Optimizer ---
    print("\n--- Initializing Model, Loss, Optimizer ---")

    # Số lượng output classes được định nghĩa cứng trong model.py
    # Chúng ta không cần truyền num_classes ở đây nữa
    # Cần đảm bảo số lớp cứng trong model.py khớp với CLASS_NAMES ở đây!
    num_classes_check = { # dictionary này chỉ dùng để kiểm tra/tham khảo
        'gender': len(GENDER_NAMES),
        'age': len(AGE_NAMES),
        'ethnicity': len(ETHNICITY_NAMES),
        'beard': len(BEARD_NAMES),
        'glasses': len(GLASSES_NAMES),
        'accessories': len(ACCESSORIES_NAMES)
    }
    # print(f"Expected number of classes: {num_classes_check}") # Có thể in ra để kiểm tra

    # Khởi tạo model mà không truyền tham số num_classes
    model = FeatureClassifier().to(device) # Bỏ tham số num_classes
    # Bạn có thể truyền freeze_backbone và dropout_rate nếu muốn
    # model = FeatureClassifier(freeze_backbone=True, dropout_rate=0.5).to(device)

    criterion = {
        feature: nn.CrossEntropyLoss() for feature in FEATURES
    }
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Thêm learning rate scheduler
    if USE_LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=LR_FACTOR, 
            patience=LR_PATIENCE, 
            min_lr=LR_MIN
        )

    # --- Vòng lặp Huấn luyện ---
    best_val_loss = float('inf')
    start_epoch = 0

    # Resume from checkpoint if exists
    if BEST_MODEL_PATH.exists():
         print(f"Found checkpoint at: {BEST_MODEL_PATH}. Loading...")
         try:
             checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
             model.load_state_dict(checkpoint['model_state_dict'])
             # Load optimizer state if needed (and if it matches)
             # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             start_epoch = checkpoint.get('epoch', 0) + 1 # Get epoch, default to 0 if not found
             best_val_loss = checkpoint.get('val_loss', float('inf')) # Get saved best loss
             print(f"Resuming training from epoch {start_epoch}, best val loss: {best_val_loss:.4f}")
         except Exception as e:
             print(f"Error loading checkpoint: {e}. Starting training from scratch.")
             start_epoch = 0
             best_val_loss = float('inf') # Reset best loss

    
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)

    
    print(f"\n--- Starting Training from Epoch {start_epoch + 1} ---")
    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # --- Training phase ---
        model.train()
        total_train_loss = 0
        # Correct predictions cho từng feature
        correct_train = {feature: 0 for feature in FEATURES}
        total_train_samples = 0 # Count valid samples processed
        batches_processed = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
        for batch_data in progress_bar:
            if batch_data is None:
                continue # Skip if collate_fn returned None
            batches_processed += 1
            
            # batch_data từ DataLoader là tuple (images, features_tensor)
            images, features_tensor = batch_data

            images = images.to(device)
            # Chuyển features_tensor thành dictionary of tensors cho từng feature
            targets = {feature: features_tensor[:, i].to(device) for i, feature in enumerate(FEATURES)}


            try:
                optimizer.zero_grad()
                outputs = model(images) # outputs là dictionary {'gender': tensor, 'age': tensor, ...}

                losses = {}
                valid_batch_for_loss = True
                for feature_name in FEATURES:
                     if feature_name in outputs and feature_name in targets:
                         if feature_name in criterion:
                              losses[feature_name] = criterion[feature_name](outputs[feature_name], targets[feature_name])
                         else:
                              warnings.warn(f"No criterion defined for feature '{feature_name}'. Skipping loss for this key.")
                              valid_batch_for_loss = False
                     else:
                          warnings.warn(f"Missing feature '{feature_name}' in model outputs or targets during training. Skipping loss for this key.")
                          valid_batch_for_loss = False

                if not losses or not valid_batch_for_loss: # Skip if no valid losses calculated
                     warnings.warn(f"No valid losses calculated for batch {progress_bar.n}. Skipping backprop.")
                     continue

                loss = sum(losses.values())

                loss.backward()
                optimizer.step()

                batch_size_actual = images.size(0)
                total_train_loss += loss.detach().item() * batch_size_actual # Use detached loss item
                total_train_samples += batch_size_actual
                
                # Tính accuracy cho từng feature
                for feature_name in FEATURES:
                    if feature_name in outputs and feature_name in targets: # Check key exists before calculating accuracy
                         _, predicted = outputs[feature_name].max(1)
                         correct_train[feature_name] += predicted.eq(targets[feature_name]).sum().item()


                # Update progress bar based on cumulative average
                current_avg_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
                # Hiển thị accuracy cho tất cả features
                acc_postfix = {f'{f.capitalize()}_acc': (100.0 * correct_train[f] / total_train_samples) if total_train_samples > 0 else 0 for f in FEATURES}
                
                progress_bar.set_postfix({
                    'avg_loss': f"{current_avg_loss:.4f}",
                    **acc_postfix # Mở rộng dictionary accuracy
                })

            except Exception as e:
                 warnings.warn(f"Error during training batch {progress_bar.n}: {e}. Skipping batch.")
                 # Clean up potentially problematic tensors
                 del images, features_tensor, targets
                 if 'outputs' in locals(): del outputs
                 if 'loss' in locals(): del loss
                 if 'losses' in locals(): del losses
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 continue

        # Calculate final epoch averages
        avg_train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
        train_accuracies = {feature: (100. * correct_train[feature] / total_train_samples) if total_train_samples > 0 else 0
                           for feature in FEATURES}
        print(f"\nEpoch {epoch+1} Training finished. Processed {total_train_samples} samples in {batches_processed} non-empty batches.")

        # --- Validation phase ---
        # Validate only if there are validation samples and training occurred
        if len(val_dataset_transformed) > 0 and total_train_samples > 0: # Sử dụng val_dataset_transformed
             avg_val_loss, val_accuracies = validate(model, val_loader, criterion, device)
        else:
             print("Skipping validation phase (no validation data or no training samples processed).")
             avg_val_loss = float('inf') # Set val_loss to infinity if skipped
             val_accuracies = {feature: 0 for feature in FEATURES} # Khởi tạo accuracies về 0

        
        if USE_LR_SCHEDULER and len(val_dataset_transformed) > 0 and total_train_samples > 0: # Sử dụng val_dataset_transformed
            print(f"Learning rate: {scheduler.optimizer.param_groups[0]['lr']}")
            scheduler.step(avg_val_loss)

        
        # Execute early stopping check only if validation was performed
        if len(val_dataset_transformed) > 0 and total_train_samples > 0:
            if early_stopping(avg_val_loss):
                print(f"\nEarly stopping triggered after epoch {epoch+1}!")
                break  # Exit the training loop


        # Print epoch results
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'  Training Loss: {avg_train_loss:.4f}')
        # In training accuracies cho tất cả features
        train_acc_str = ', '.join([f'{f.capitalize()}={train_accuracies[f]:.2f}%' for f in FEATURES])
        print(f'  Training Accuracies: {train_acc_str}')

        print(f'  Validation Loss: {avg_val_loss:.4f}')
        # In validation accuracies cho tất cả features
        val_acc_str = ', '.join([f'{f.capitalize()}={val_accuracies[f]:.2f}%' for f in FEATURES])
        print(f'  Validation Accuracies: {val_acc_str}')


        # Save best model based on validation loss
        if avg_val_loss < best_val_loss and total_train_samples > 0 and avg_val_loss > 0:
            best_val_loss = avg_val_loss
            print(f'  ** New best validation loss: {best_val_loss:.4f}. Saving model to {BEST_MODEL_PATH}... **')
            try:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies
                }, BEST_MODEL_PATH)
                print("   Model saved successfully.")
            except Exception as e:
                 print(f"  Error saving model: {e}")
        elif total_train_samples == 0:
             print("  Skipping model saving: No valid training samples processed.")
        elif avg_val_loss <= 0: # Check for non-positive val loss as well
             print("  Skipping model saving: Non-positive validation loss (potential issue).")
        else:
             print(f'  Validation loss ({avg_val_loss:.4f}) did not improve from best ({best_val_loss:.4f}).')


    print("\n--- Training Finished ---")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")


if __name__ == '__main__':
    train()