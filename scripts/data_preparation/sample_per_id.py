"""
Giảm kích thước tập dữ liệu bằng cách lấy mẫu một số lượng ảnh nhất định cho mỗi person_id.
Chức năng:
- Đọc danh sách đường dẫn ảnh từ một file text đầu vào (thường là output của bước lọc trước đó).
- Trích xuất person_id từ mỗi đường dẫn ảnh dựa trên cấu trúc thư mục được chỉ định (ví dụ: thư mục cha).
- Nhóm các đường dẫn ảnh theo person_id.
- Với mỗi person_id, lấy mẫu ngẫu nhiên một số lượng ảnh tối đa được cấu hình (MAX_IMAGES_PER_ID).
  Nếu số lượng ảnh của một ID nhỏ hơn hoặc bằng giới hạn, tất cả ảnh của ID đó sẽ được giữ lại.
- Lưu danh sách đường dẫn ảnh đã được lấy mẫu vào một file text mới.
- Hữu ích để giảm số lượng ảnh cần xử lý/huấn luyện mà vẫn duy trì sự đa dạng về ID.
"""
# scripts/data_preparation/sample_per_id.py
import random
from pathlib import Path
from collections import defaultdict, Counter
import os
import numpy as np # Thêm numpy để tính toán thống kê
from tqdm import tqdm # Thêm tqdm để hiển thị tiến trình
import csv # Thêm thư viện csv để ghi file thống kê

# --- Configuration ---
INPUT_FILE_PATH = Path("final_content_filtered.txt")  # File chứa đường dẫn ảnh sau khi lọc YOLO
OUTPUT_FILE_PATH = Path("final_sampled.txt") # File output chứa đường dẫn sau khi lấy mẫu
# !!! File CSV để lưu thống kê chi tiết số lượng ảnh của từng ID !!!
STATS_OUTPUT_CSV_PATH = Path("id_counts_statistics.csv")
MAX_IMAGES_PER_ID = 500 # Số lượng ảnh tối đa muốn giữ lại cho mỗi person_id
ID_EXTRACTION_LEVEL = -2 # Mức thư mục chứa ID ( -1 là tên file, -2 là thư mục cha, -3 là thư mục ông...)

# --- Main Logic ---
if not INPUT_FILE_PATH.exists():
    print(f"Error: Input file not found at '{INPUT_FILE_PATH}'")
    exit()

print(f"Reading paths from: '{INPUT_FILE_PATH}'")
image_paths = []
try:
    # Ước tính số dòng để tqdm hiển thị tiến trình tốt hơn (tùy chọn)
    try:
        total_lines = sum(1 for line in open(INPUT_FILE_PATH, 'r'))
        print(f"Estimated lines in input file: {total_lines}")
    except Exception:
        total_lines = None # Không ước tính được

    with open(INPUT_FILE_PATH, 'r') as f:
        # Thêm tqdm nếu file lớn
        pbar = tqdm(f, total=total_lines, desc="Reading input file", unit=" lines") if total_lines else f
        for line in pbar:
            cleaned_line = line.strip()
            if cleaned_line and not cleaned_line.startswith('#'):
                image_paths.append(cleaned_line)
    print(f"Read {len(image_paths)} total image paths.")
except Exception as e:
    print(f"Error reading input file '{INPUT_FILE_PATH}': {e}")
    exit()

if not image_paths:
    print("Input file is empty.")
    exit()

# Nhóm ảnh theo person_id
images_by_id = defaultdict(list)
extraction_errors = 0
print(f"Grouping images by person_id (assuming ID is at level {ID_EXTRACTION_LEVEL} in the path)...")

for path_str in tqdm(image_paths, desc="Grouping images"):
    try:
        parts = Path(path_str).parts
        if len(parts) > abs(ID_EXTRACTION_LEVEL):
            person_id = parts[ID_EXTRACTION_LEVEL]
            images_by_id[person_id].append(path_str)
        else:
            # print(f"Warning: Could not extract ID from path '{path_str}' with level {ID_EXTRACTION_LEVEL}. Skipping.")
            extraction_errors += 1
    except IndexError:
        # print(f"Warning: Index error extracting ID from path '{path_str}' with level {ID_EXTRACTION_LEVEL}. Skipping.")
        extraction_errors += 1
    except Exception as e:
        # print(f"Warning: Unexpected error processing path '{path_str}': {e}. Skipping.")
        extraction_errors += 1


num_unique_ids = len(images_by_id)
print(f"\nFound {num_unique_ids} unique person_ids.")
if extraction_errors > 0:
    print(f"Skipped {extraction_errors} paths due to ID extraction issues.")

# --- Thống kê số lượng ảnh trên mỗi ID ---
print("\n--- Statistics Before Sampling ---")
counts_per_id_dict = {} # Lưu trữ {person_id: count}
if num_unique_ids > 0:
    # Tính toán số lượng cho từng ID
    for person_id, paths in images_by_id.items():
        counts_per_id_dict[person_id] = len(paths)

    counts_list = list(counts_per_id_dict.values()) # List các số lượng để tính thống kê
    min_count = np.min(counts_list)
    max_count = np.max(counts_list)
    avg_count = np.mean(counts_list)
    median_count = np.median(counts_list)
    std_dev_count = np.std(counts_list)

    print(f"Total unique person_ids: {num_unique_ids}")
    print(f"Total images processed: {len(image_paths)}") # Tổng số ảnh đọc được
    print("\nSummary - Images per person_id:")
    print(f"  Min: {min_count}")
    print(f"  Max: {max_count}")
    print(f"  Average: {avg_count:.2f}")
    print(f"  Median: {median_count}")
    print(f"  Standard Deviation: {std_dev_count:.2f}")

    # Thống kê phân phối (hiển thị trên console)
    print("\nSummary - Distribution of image counts per ID:")
    count_distribution = Counter(counts_list)
    max_val = max_count
    step = 10 # Khoảng cách các bin, có thể điều chỉnh
    bins = list(range(1, max_val + step, step))
    hist = defaultdict(int)
    ids_in_bin = defaultdict(int)

    for count_val, num_ids_with_that_count in count_distribution.items():
        for i in range(len(bins) - 1):
             if bins[i] <= count_val < bins[i+1]:
                 bin_label = f"{bins[i]}-{bins[i+1]-1}"
                 hist[bin_label] += num_ids_with_that_count * count_val
                 ids_in_bin[bin_label] += num_ids_with_that_count
                 break
        else:
             bin_label = f"{bins[-1] if bins else 1}+" # Xử lý nếu chỉ có 1 bin
             hist[bin_label] += num_ids_with_that_count * count_val
             ids_in_bin[bin_label] += num_ids_with_that_count

    sorted_bins = sorted(ids_in_bin.keys(), key=lambda x: int(x.split('-')[0].replace('+','')))
    for bin_label in sorted_bins:
         num_ids = ids_in_bin[bin_label]
         total_images_in_bin = hist[bin_label]
         print(f"  {bin_label} images: {num_ids} IDs (totaling {total_images_in_bin} images)")

    # --- Lưu thống kê chi tiết của từng ID vào file CSV ---
    print(f"\nSaving detailed image counts per ID to: '{STATS_OUTPUT_CSV_PATH.resolve()}'...")
    try:
        with open(STATS_OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['person_id', 'image_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            # Sắp xếp theo ID trước khi ghi (tùy chọn)
            sorted_ids = sorted(counts_per_id_dict.keys())
            for person_id in tqdm(sorted_ids, desc="Writing statistics file"):
                writer.writerow({'person_id': person_id, 'image_count': counts_per_id_dict[person_id]})
        print("Detailed statistics saved successfully.")
    except Exception as e:
        print(f"Error saving statistics CSV file: {e}")

else:
    print("No valid person_ids found to calculate statistics.")

# --- Lấy mẫu ngẫu nhiên cho mỗi ID ---
sampled_image_paths = []
print(f"\n--- Sampling up to {MAX_IMAGES_PER_ID} images per person_id ---")

for person_id, paths in tqdm(images_by_id.items(), desc="Sampling"):
    if len(paths) > MAX_IMAGES_PER_ID:
        sampled_paths = random.sample(paths, MAX_IMAGES_PER_ID)
        sampled_image_paths.extend(sampled_paths)
    else:
        sampled_image_paths.extend(paths)

print(f"\nTotal images after sampling: {len(sampled_image_paths)}")

# --- Lưu kết quả vào file mới ---
print(f"Saving sampled paths to: '{OUTPUT_FILE_PATH.resolve()}'...")
try:
    with open(OUTPUT_FILE_PATH, 'w') as f:
        sampled_image_paths.sort()
        for path_str in sampled_image_paths:
            f.write(f"{path_str}\n")
    print("Save complete.")
except Exception as e:
    print(f"Error saving output file: {e}")

print("\nSampling and Statistics script finished.")
