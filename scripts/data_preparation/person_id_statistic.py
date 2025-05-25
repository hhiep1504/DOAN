import csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# --- Configuration ---
STATS_CSV_PATH = Path("id_counts_statistics.csv") # Đường dẫn đến file CSV thống kê

# --- Main Logic ---
if not STATS_CSV_PATH.exists():
    print(f"Lỗi: Không tìm thấy file thống kê tại '{STATS_CSV_PATH}'")
    exit()

print(f"Đang đọc và phân tích file: '{STATS_CSV_PATH}'")

counts_per_id_dict = {}
image_counts_list = []

try:
    with open(STATS_CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        if 'person_id' not in reader.fieldnames or 'image_count' not in reader.fieldnames:
             print(f"Lỗi: File CSV phải chứa cột 'person_id' và 'image_count'")
             exit()
        for row in reader:
            try:
                person_id = row['person_id']
                count = int(row['image_count'])
                if count < 0:
                    print(f"Cảnh báo: Bỏ qua ID '{person_id}' với số lượng ảnh âm: {count}")
                    continue
                counts_per_id_dict[person_id] = count
                image_counts_list.append(count)
            except ValueError:
                print(f"Cảnh báo: Bỏ qua dòng không hợp lệ trong CSV: {row}")
            except Exception as e:
                 print(f"Cảnh báo: Lỗi không mong muốn khi xử lý dòng {row}: {e}")

except Exception as e:
    print(f"Lỗi khi đọc file CSV '{STATS_CSV_PATH}': {e}")
    exit()

num_unique_ids = len(counts_per_id_dict)

if num_unique_ids == 0:
    print("Không tìm thấy dữ liệu hợp lệ trong file CSV.")
    exit()

print(f"\n--- Thống kê chi tiết từ File ---")
print(f"Tổng số person_id duy nhất: {num_unique_ids}")
print(f"Tổng số ảnh (từ file CSV): {sum(image_counts_list)}") # Tổng số ảnh dựa trên file csv

# --- Tính toán thống kê cơ bản ---
if image_counts_list:
    min_count = np.min(image_counts_list)
    max_count = np.max(image_counts_list)
    avg_count = np.mean(image_counts_list)
    median_count = np.median(image_counts_list)
    std_dev_count = np.std(image_counts_list)

    print("\nThống kê tóm tắt - Số lượng ảnh trên mỗi person_id:")
    print(f"  Min: {min_count}")
    print(f"  Max: {max_count}")
    print(f"  Average (Trung bình): {avg_count:.2f}")
    print(f"  Median (Trung vị): {median_count}")
    print(f"  Standard Deviation (Độ lệch chuẩn): {std_dev_count:.2f}")

    # --- Thống kê phân phối theo khoảng ---
    print("\nThống kê tóm tắt - Phân phối số lượng ảnh theo ID:")
    count_distribution = Counter(image_counts_list)
    # Định nghĩa các khoảng (bins) bạn muốn xem
    # Ví dụ: <100, 100-499, 500-2499, 2500-4999, >=5000
    bins = [(0, 99), (100, 499), (500, 2499), (2500, 4999), (5000, float('inf'))]
    distribution_summary = defaultdict(lambda: {'count': 0, 'total_images': 0})

    for count_val, num_ids_with_that_count in count_distribution.items():
        assigned = False
        for lower, upper in bins:
            if lower <= count_val <= upper:
                if upper == float('inf'):
                    label = f"{lower}+"
                else:
                    label = f"{lower}-{upper}"
                distribution_summary[label]['count'] += num_ids_with_that_count
                distribution_summary[label]['total_images'] += num_ids_with_that_count * count_val
                assigned = True
                break
        if not assigned: # Trường hợp đặc biệt nếu có giá trị 0 hoặc âm (đã lọc ở trên nhưng để phòng ngừa)
             print(f"Cảnh báo: Giá trị count {count_val} không thuộc khoảng nào.")


    # Sắp xếp các khoảng để in theo thứ tự
    def sort_key(bin_label):
        if '+' in bin_label:
            return int(bin_label.replace('+', ''))
        return int(bin_label.split('-')[0])

    sorted_bins = sorted(distribution_summary.keys(), key=sort_key)

    for bin_label in sorted_bins:
         num_ids = distribution_summary[bin_label]['count']
         total_images_in_bin = distribution_summary[bin_label]['total_images']
         print(f"  Khoảng {bin_label} ảnh: {num_ids} ID (tổng cộng {total_images_in_bin} ảnh)")

else:
    print("Danh sách số lượng ảnh trống, không thể tính toán thống kê.")

print("\nPhân tích hoàn tất.")