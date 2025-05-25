import torch
from torch.utils.data import ConcatDataset, DataLoader 
from scripts.infer.dataset_local import PdestreFeatureDataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import os
import random 
from tqdm import tqdm

# --- Định nghĩa tên các lớp cho features mới ---
GENDER_NAMES = {0: 'Nam', 1: 'Nữ', 2: 'Không xác định', -1: 'Thiếu/Lỗi'}
AGE_NAMES = {
    0: '0-17', 1: '18-54', 2: '55+', 3: 'Không xác định', -1: 'Thiếu/Lỗi'
}
ETHNICITY_NAMES = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Không xác định', -1: 'Thiếu/Lỗi'}
BEARD_NAMES = {0: 'Có râu', 1: 'Không râu', 2: 'Không rõ', -1: 'Thiếu/Lỗi'}
GLASSES_NAMES = {0: 'Kính thường', 1: 'Kính râm', 2: 'Không kính', 3: 'Không rõ', -1: 'Thiếu/Lỗi'}

# --- Giữ nguyên hàm visualize_sample ---
def visualize_sample(image, features):
    """Visualize một sample từ dataset"""
    # Chuyển tensor image thành numpy array và denormalize
    # Cần lấy mean/std từ transform (hoặc dùng giá trị mặc định nếu biết chắc)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    try:
        image_np = image.numpy().transpose((1, 2, 0))
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
    except Exception as e:
        print(f"Error during image denormalization/conversion: {e}")
        # Tạo ảnh placeholder màu xám nếu lỗi
        image_np = np.ones((224, 224, 3)) * 0.5

    # Tạo figure
    plt.figure(figsize=(10, 5))

    # Hiển thị ảnh
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.axis('off')
    plt.title('Image')

    # Hiển thị features
    plt.subplot(1, 2, 2)
    # Lấy giá trị từ tensor features
    try:
        gender_val = features['gender'].item() if 'gender' in features else 'N/A'
        age_val = features['age'].item() if 'age' in features else 'N/A'
        ethnicity_val = features['ethnicity'].item() if 'ethnicity' in features else 'N/A'
        beard_val = features['beard'].item() if 'beard' in features else 'N/A'
        glasses_val = features['glasses'].item() if 'glasses' in features else 'N/A'

        feature_text = f"Gender: {GENDER_NAMES.get(gender_val, gender_val)}\n"
        feature_text += f"Age: {AGE_NAMES.get(age_val, age_val)}\n"
        feature_text += f"Ethnicity: {ETHNICITY_NAMES.get(ethnicity_val, ethnicity_val)}\n"
        feature_text += f"Beard: {BEARD_NAMES.get(beard_val, beard_val)}\n"
        feature_text += f"Glasses: {GLASSES_NAMES.get(glasses_val, glasses_val)}"
    except Exception as e:
        print(f"Error accessing features: {e}")
        feature_text = "Error loading features"

    plt.text(0.1, 0.5, feature_text, fontsize=12, va='center') # Căn giữa theo chiều dọc
    plt.axis('off')
    plt.title('Features')

    plt.tight_layout()
    plt.show()

# --- Hàm check_dataset ---
def check_combined_dataset(jpg_dir_path, annotation_dir_path, num_samples_to_check=100, num_samples_to_show=5):
    """Kiểm tra và thống kê dataset kết hợp từ nhiều file annotation"""

    jpg_dir = Path(jpg_dir_path)
    annotation_dir = Path(annotation_dir_path)

    # --- 1. Kiểm tra sự tồn tại của các thư mục chính ---
    print("--- Phase 1: Checking Base Directories ---")
    if not jpg_dir.exists():
        print(f"Error: Image directory '{jpg_dir}' does not exist!")
        return
    if not annotation_dir.exists():
        print(f"Error: Annotation directory '{annotation_dir}' does not exist!")
        return
    print(f"Found Image Directory: {jpg_dir}")
    print(f"Found Annotation Directory: {annotation_dir}")

    # --- 2. Tìm và tải các dataset con ---
    print("\n--- Phase 2: Loading Datasets from Annotations ---")
    all_annotation_files = sorted(list(annotation_dir.glob('*.txt')))

    if not all_annotation_files:
        print(f"Error: No .txt annotation files found in {annotation_dir}")
        return
    print(f"Found {len(all_annotation_files)} potential annotation files.")

    list_of_datasets = []
    failed_files = []
    # Định nghĩa transform dùng chung
    common_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    for ann_file in tqdm(all_annotation_files, desc="Loading individual datasets"):
        try:
            dataset_instance = PdestreFeatureDataset(
                jpg_dir=jpg_dir,
                annotation_file=ann_file,
                transform=common_transform
            )
            if len(dataset_instance) > 0:
                list_of_datasets.append(dataset_instance)
            # else: # Không cần báo nếu file annotation rỗng nhưng hợp lệ
            #     print(f"Info: Annotation file {ann_file.name} resulted in an empty dataset (possibly all filtered out).")
        except Exception as e:
            print(f"\nWarning: Failed to load dataset from {ann_file.name}. Error: {e}")
            failed_files.append(ann_file.name)

    if not list_of_datasets:
        print("\nError: Could not load any valid datasets from the annotation files.")
        if failed_files:
            print("Failed files:", failed_files)
        return

    print(f"\nSuccessfully loaded {len(list_of_datasets)} non-empty dataset(s).")
    if failed_files:
        print(f"Skipped {len(failed_files)} file(s) due to errors: {failed_files}")

    # --- 3. Tạo dataset kết hợp ---
    combined_dataset = ConcatDataset(list_of_datasets)
    total_samples = len(combined_dataset)
    print(f"Total combined dataset size: {total_samples} samples.")

    # --- 4. Kiểm tra cấu trúc thư mục ảnh (tương tự như trước) ---
    print("\n--- Phase 3: Checking Image Directory Structure (Sample) ---")
    date_dirs = [d for d in jpg_dir.iterdir() if d.is_dir()]
    print(f"Found {len(date_dirs)} date directories in {jpg_dir}")
    if date_dirs:
        print("Checking first date directory:")
        d = date_dirs[0] # Chỉ kiểm tra thư mục ngày đầu tiên cho ngắn gọn
        print(f"- Directory: {d.name}")
        id_dirs = [id_dir for id_dir in d.iterdir() if id_dir.is_dir()]
        print(f"  Contains {len(id_dirs)} ID folders")
        if id_dirs:
            print("  Checking first ID folder:")
            id_dir = id_dirs[0]
            print(f"    - Folder: {id_dir.name}")
            images = list(id_dir.glob("*.jpg"))
            print(f"      Contains {len(images)} images")
            if images:
                print(f"      Example image file: {images[0].name}")
        else:
            print("  No ID folders found in this date directory.")
    else:
        print("No date directories found inside the main image directory.")


    # --- 5. Thống kê features và kiểm tra lỗi trên các mẫu ---
    print(f"\n--- Phase 4: Checking first {min(num_samples_to_check, total_samples)} samples ---")
    gender_counts = {0: 0, 1: 0, 2: 0, -1: 0} # Thêm -1 cho lỗi/thiếu
    age_counts = {i: 0 for i in range(4)}
    age_counts[-1] = 0 # Thêm -1 cho lỗi/thiếu
    ethnicity_counts = {i: 0 for i in range(5)}
    ethnicity_counts[-1] = 0 # Thêm -1 cho lỗi/thiếu
    beard_counts = {i: 0 for i in range(3)}
    beard_counts[-1] = 0 # Cho giá trị thiếu/lỗi
    glasses_counts = {i: 0 for i in range(4)}
    glasses_counts[-1] = 0 # Cho giá trị thiếu/lỗi

    error_indices = []
    valid_samples_count = 0
    checked_count = 0

    # Lặp qua một phần hoặc toàn bộ dataset kết hợp
    indices_to_check = range(min(num_samples_to_check, total_samples))

    for i in tqdm(indices_to_check, desc="Checking samples"):
        checked_count += 1
        try:
            # Lấy mẫu trực tiếp từ combined_dataset
            image, features = combined_dataset[i]

            # Kiểm tra dữ liệu trả về hợp lệ không
            if image is None or features is None:
                 raise ValueError("Dataset returned None for image or features.")

            # Cập nhật thống kê nếu hợp lệ
            gender_val = features['gender'].item()
            age_val = features['age'].item()
            ethnicity_val = features['ethnicity'].item()
            beard_val = features['beard'].item()
            glasses_val = features['glasses'].item()

            gender_counts[gender_val] += 1
            # Đảm bảo age và ethnicity nằm trong khoảng mong đợi trước khi cập nhật
            if age_val in age_counts:
                age_counts[age_val] += 1
            else:
                age_counts[-1] += 1 # Coi như lỗi nếu ngoài khoảng 0-3
            if ethnicity_val in ethnicity_counts:
                ethnicity_counts[ethnicity_val] += 1
            else:
                ethnicity_counts[-1] += 1 # Coi như lỗi nếu ngoài khoảng 0-4
            if beard_val in beard_counts:
                beard_counts[beard_val] += 1
            else:
                beard_counts[-1] += 1 # Lỗi nếu giá trị ngoài khoảng mong đợi
            if glasses_val in glasses_counts:
                glasses_counts[glasses_val] += 1
            else:
                glasses_counts[-1] += 1 # Lỗi nếu giá trị ngoài khoảng mong đợi

            # Kiểm tra kích thước ảnh
            if not isinstance(image, torch.Tensor) or image.shape != (3, 224, 224):
                print(f"\nWarning: Sample {i} has incorrect image type/shape: {type(image)}, {image.shape if hasattr(image, 'shape') else 'N/A'}")
                # Vẫn có thể tính là mẫu hợp lệ nếu features ổn, tùy quan điểm
                # error_indices.append(i) # Bỏ vào lỗi nếu muốn loại bỏ hoàn toàn
                # continue # Bỏ qua nếu shape sai

            valid_samples_count += 1

        except Exception as e:
            # Ghi nhận lỗi cho sample index i
            print(f"\nError processing sample at combined index {i}: {e}")
            error_indices.append(i)

    # --- 6. In thống kê ---
    print("\n--- Phase 5: Statistics Summary ---")
    print(f"Checked {checked_count} samples.")
    print(f"Valid samples found (among checked): {valid_samples_count}")
    print(f"Errors encountered (among checked): {len(error_indices)}")
    if error_indices:
         print(f"  Example error indices: {error_indices[:10]}...") # In ra vài index lỗi đầu tiên

    if valid_samples_count > 0:
        print(f"\nFeature distribution (based on {valid_samples_count} valid checked samples):")

        print("\nGender distribution:")
        for gender, count in gender_counts.items():
            if count > 0: # Chỉ in những category có dữ liệu
                label = GENDER_NAMES.get(gender, f"Unknown_Code:{gender}")
                print(f"  Gender {label}: {count} samples ({count/valid_samples_count*100:.2f}%)")

        print("\nAge distribution:")
        for age, count in age_counts.items():
             if count > 0:
                label = AGE_NAMES.get(age, f"Unknown_Code:{age}")
                print(f"  Age {label}: {count} samples ({count/valid_samples_count*100:.2f}%)")

        print("\nEthnicity distribution:")
        for ethnicity, count in ethnicity_counts.items():
            if count > 0:
                label = ETHNICITY_NAMES.get(ethnicity, f"Unknown_Code:{ethnicity}")
                print(f"  Ethnicity {label}: {count} samples ({count/valid_samples_count*100:.2f}%)")

        print("\nBeard distribution:")
        for beard, count in beard_counts.items():
            if count > 0:
                label = BEARD_NAMES.get(beard, f"Unknown_Code:{beard}")
                print(f"  Beard {label}: {count} samples ({count/valid_samples_count*100:.2f}%)")

        print("\nGlasses distribution:")
        for glasses, count in glasses_counts.items():
            if count > 0:
                label = GLASSES_NAMES.get(glasses, f"Unknown_Code:{glasses}")
                print(f"  Glasses {label}: {count} samples ({count/valid_samples_count*100:.2f}%)")
    else:
        print("\nNo valid samples found among the checked ones to calculate distributions.")

    # --- 7. Visualize một số mẫu hợp lệ ngẫu nhiên ---
    print(f"\n--- Phase 6: Visualizing {num_samples_to_show} Random Valid Samples ---")
    # Lấy danh sách các index hợp lệ từ những mẫu đã kiểm tra
    possible_indices = list(set(indices_to_check) - set(error_indices))

    if not possible_indices:
        print("No valid samples available to visualize.")
        return

    # Chọn ngẫu nhiên từ các index hợp lệ đã biết
    num_to_actually_show = min(num_samples_to_show, len(possible_indices))
    indices_to_show = random.sample(possible_indices, num_to_actually_show)
    print(f"Selected indices to visualize: {indices_to_show}")

    for count, idx in enumerate(indices_to_show):
        print(f"\nVisualizing sample {count+1}/{num_to_actually_show} (Combined Index: {idx})")
        try:
            image, features = combined_dataset[idx]
            if image is not None and features is not None:
                 visualize_sample(image, features)
            else:
                 print("  Skipping visualization because data returned was None.")
        except Exception as e:
            print(f"  Error visualizing sample at index {idx}: {e}")


if __name__ == '__main__':
    # !!! THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY CHO PHÙ HỢP VỚI MÁY CỦA BẠN !!!
    JPG_DIR = 'data/raw/jpg_Extracted_PIDS'
    ANNOTATION_DIR = 'data/raw/P-DESTRE/annotation'

    # Cấu hình số lượng mẫu kiểm tra và hiển thị
    NUM_SAMPLES_TO_CHECK = 50 # Kiểm tra 500 mẫu đầu tiên để lấy thống kê
    NUM_SAMPLES_TO_SHOW = 5   # Hiển thị 5 mẫu ngẫu nhiên hợp lệ

    check_combined_dataset(
        jpg_dir_path=JPG_DIR,
        annotation_dir_path=ANNOTATION_DIR,
        num_samples_to_check=NUM_SAMPLES_TO_CHECK,
        num_samples_to_show=NUM_SAMPLES_TO_SHOW
    )
    print("\nDataset check finished.")