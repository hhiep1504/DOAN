import torch
from torch.utils.data import ConcatDataset, random_split # Thêm random_split
# Giả sử PdestreFeatureDataset được định nghĩa trong dataset.py
try:
    from scripts.infer.dataset_local import PdestreFeatureDataset
except ImportError:
    print("Error: Make sure 'dataset.py' with PdestreFeatureDataset class is in the same directory.")
    exit()

from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import math

# ======================================================================
# Configuration
# ======================================================================
# !!! THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY CHO PHÙ HỢP VỚI MÁY CỦA BẠN !!!
JPG_DIR = Path("jpg_Extracted_PIDS")  # Đường dẫn đến thư mục chứa các thư mục con theo ngày
ANNOTATION_DIR = Path("P-DESTRE/annotation") # Đường dẫn đến thư mục chứa các file .txt

# --- Tỉ lệ chia Dataset ---
# Ví dụ: 80% cho training, 20% cho validation
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
# TEST_RATIO = 0.0 # Đặt > 0 nếu bạn muốn có tập test riêng (ví dụ: 0.7, 0.15, 0.15)

# --- Random Seed ---
# Đặt một seed cố định để việc chia là nhất quán mỗi lần chạy (tùy chọn)
RANDOM_SEED = 42
if RANDOM_SEED:
    torch.manual_seed(RANDOM_SEED)

# ======================================================================
# Main execution block
# ======================================================================
if __name__ == "__main__":

    print("--- Starting Dataset Loading and Splitting ---")

    # --- 1. Kiểm tra sự tồn tại của các thư mục chính ---
    print("\n--- Phase 1: Checking Base Directories ---")
    if not JPG_DIR.exists():
        print(f"Error: Image directory '{JPG_DIR}' does not exist!")
        exit()
    if not ANNOTATION_DIR.exists():
        print(f"Error: Annotation directory '{ANNOTATION_DIR}' does not exist!")
        exit()
    print(f"Found Image Directory: {JPG_DIR}")
    print(f"Found Annotation Directory: {ANNOTATION_DIR}")

    # --- 2. Tìm và tải các dataset con ---
    print("\n--- Phase 2: Loading Datasets from Annotations ---")
    all_annotation_files = sorted(list(ANNOTATION_DIR.glob('*.txt')))

    if not all_annotation_files:
        print(f"Error: No .txt annotation files found in {ANNOTATION_DIR}")
        exit()
    print(f"Found {len(all_annotation_files)} potential annotation files.")

    list_of_datasets = []
    failed_files = []
    # Định nghĩa transform dùng chung (có thể giống hệt trong quá trình train)
    common_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    for ann_file in tqdm(all_annotation_files, desc="Loading individual datasets"):
        try:
            dataset_instance = PdestreFeatureDataset(
                jpg_dir=JPG_DIR,
                annotation_file=ann_file,
                transform=common_transform
            )
            if len(dataset_instance) > 0:
                list_of_datasets.append(dataset_instance)
        except Exception as e:
            print(f"\nWarning: Failed to load dataset from {ann_file.name}. Error: {e}")
            failed_files.append(ann_file.name)

    if not list_of_datasets:
        print("\nError: Could not load any valid datasets from the annotation files.")
        exit()

    print(f"\nSuccessfully loaded {len(list_of_datasets)} non-empty dataset(s).")
    if failed_files:
        print(f"Skipped {len(failed_files)} file(s) due to errors: {failed_files}")

    # --- 3. Tạo dataset kết hợp ---
    combined_dataset = ConcatDataset(list_of_datasets)
    total_samples = len(combined_dataset)
    print(f"\nTotal combined dataset size: {total_samples} samples.")

    # --- 4. Chia Dataset ---
    print("\n--- Phase 3: Splitting Dataset ---")

    if total_samples == 0:
        print("Error: Combined dataset is empty, cannot split.")
        exit()

    # Tính toán số lượng mẫu cho mỗi tập
    train_size = int(TRAIN_RATIO * total_samples)
    val_size = int(VAL_RATIO * total_samples)
    # test_size = int(TEST_RATIO * total_samples) # Bật nếu dùng TEST_RATIO

    # Điều chỉnh để đảm bảo tổng số lượng khớp (quan trọng!)
    # Nếu chỉ có train/val:
    current_sum = train_size + val_size
    if current_sum != total_samples:
        # Thường cộng phần dư vào tập train
        train_size += (total_samples - current_sum)
        print(f"Adjusting split sizes slightly: Train={train_size}, Val={val_size}")

    # # Nếu có cả train/val/test:
    # current_sum = train_size + val_size + test_size
    # if current_sum != total_samples:
    #     train_size += (total_samples - current_sum) # Cộng phần dư vào tập train
    #     print(f"Adjusting split sizes slightly: Train={train_size}, Val={val_size}, Test={test_size}")


    print(f"Attempting to split into: Train={train_size}, Validation={val_size}")
    # print(f"Attempting to split into: Train={train_size}, Validation={val_size}, Test={test_size}") # Nếu dùng test

    try:
        # Thực hiện chia
        # Chỉ train/val:
        train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

        # # Nếu có cả train/val/test:
        # train_dataset, val_dataset, test_dataset = random_split(
        #     combined_dataset, [train_size, val_size, test_size]
        # )

        print("Dataset split successfully.")

    except Exception as e:
        print(f"\nError during dataset split: {e}")
        print("Check if total samples is positive and ratios add up correctly.")
        exit()

    # --- 5. Hiển thị kết quả ---
    print("\n--- Phase 4: Resulting Dataset Sizes ---")
    print(f"Total samples: {total_samples}")
    print(f"Training set size: {len(train_dataset)} samples ({len(train_dataset)/total_samples*100:.2f}%)")
    print(f"Validation set size: {len(val_dataset)} samples ({len(val_dataset)/total_samples*100:.2f}%)")
    # if TEST_RATIO > 0: # Chỉ in nếu có tập test
    #     print(f"Test set size: {len(test_dataset)} samples ({len(test_dataset)/total_samples*100:.2f}%)")

    print("\n--- Script Finished ---")

    # Bây giờ bạn có thể sử dụng train_dataset và val_dataset (và test_dataset nếu có)
    # để tạo DataLoader cho quá trình training và evaluation. Ví dụ:
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)