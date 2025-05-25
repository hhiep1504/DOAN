# scripts/copy_sampled_data.py
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- Cấu hình ---
# Đường dẫn đến file annotation gốc (kết quả từ script sample_per_id.py)
SOURCE_ANNOTATION_FILE = Path("final_sampled.txt")

# Đường dẫn đến file annotation mới bạn muốn tạo (bản sao)
# !!! Thay đổi đường dẫn và tên file này nếu cần !!!
DESTINATION_ANNOTATION_FILE = Path("annotations/sampled_list.txt")

# Đường dẫn đến thư mục mới bạn muốn chứa các ảnh đã sao chép
# !!! Thay đổi đường dẫn thư mục này nếu cần !!!
DESTINATION_IMAGE_DIR = Path("data/processed/images")

# !!! Thêm dòng này để xác định thư mục gốc chứa ảnh gốc !!!
SOURCE_IMAGE_BASE_DIR = Path("data/raw/jpg_Extracted_PIDS")

# Giả định đường dẫn trong SOURCE_ANNOTATION_FILE là tương đối với SOURCE_IMAGE_BASE_DIR
WORKSPACE_ROOT = Path(".") # Vẫn có thể cần nếu SOURCE_IMAGE_BASE_DIR cũng là tương đối

# --- Logic chính ---

def copy_data():
    """Sao chép file annotation và các file ảnh tương ứng."""

    # --- 1. Kiểm tra file annotation nguồn ---
    if not SOURCE_ANNOTATION_FILE.is_file():
        print(f"Lỗi: Không tìm thấy file annotation nguồn tại '{SOURCE_ANNOTATION_FILE}'.")
        return

    # --- 2. Tạo thư mục đích nếu chưa tồn tại ---
    # Tạo thư mục cho file annotation đích
    DESTINATION_ANNOTATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Tạo thư mục cho các ảnh đích
    DESTINATION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Đã tạo (hoặc đã tồn tại) thư mục annotation đích: '{DESTINATION_ANNOTATION_FILE.parent}'")
    print(f"Đã tạo (hoặc đã tồn tại) thư mục ảnh đích: '{DESTINATION_IMAGE_DIR}'")

    # --- 3. Sao chép file annotation ---
    try:
        shutil.copy2(SOURCE_ANNOTATION_FILE, DESTINATION_ANNOTATION_FILE) # copy2 giữ lại metadata
        print(f"Đã sao chép file annotation vào '{DESTINATION_ANNOTATION_FILE}'")
    except Exception as e:
        print(f"Lỗi khi sao chép file annotation: {e}")
        return # Dừng lại nếu không copy được file annotation

    # --- 4. Đọc đường dẫn ảnh và sao chép ảnh ---
    print("\nBắt đầu sao chép các file ảnh...")
    copied_count = 0
    error_count = 0
    image_paths_to_copy = []

    try:
        with open(SOURCE_ANNOTATION_FILE, 'r') as f:
            image_paths_to_copy = [line.strip() for line in f if line.strip()]
        print(f"Đọc được {len(image_paths_to_copy)} đường dẫn ảnh từ '{SOURCE_ANNOTATION_FILE}'.")
    except Exception as e:
        print(f"Lỗi khi đọc file annotation nguồn '{SOURCE_ANNOTATION_FILE}': {e}")
        return

    # Sử dụng tqdm để xem tiến trình
    for img_relative_path_str in tqdm(image_paths_to_copy, desc="Sao chép ảnh"):
        try:
            # --- !!! THAY ĐỔI Ở ĐÂY !!! ---
            # Ghép nối đường dẫn gốc của ảnh với đường dẫn tương đối đọc từ file
            source_image_path = WORKSPACE_ROOT / SOURCE_IMAGE_BASE_DIR / Path(img_relative_path_str)
            # -----------------------------

            # Kiểm tra xem file nguồn có tồn tại không
            if not source_image_path.is_file():
                print(f"\nCảnh báo: Không tìm thấy file ảnh nguồn '{source_image_path}'. Bỏ qua.")
                error_count += 1
                continue

            # Tạo đường dẫn đích
            # Giữ nguyên cấu trúc thư mục con từ img_relative_path_str
            relative_path_for_dest = Path(img_relative_path_str)
            dest_image_path = DESTINATION_IMAGE_DIR / relative_path_for_dest

            # Tạo các thư mục cha cần thiết trong thư mục đích
            dest_image_path.parent.mkdir(parents=True, exist_ok=True)

            # Thực hiện sao chép
            shutil.copy2(source_image_path, dest_image_path)
            copied_count += 1

        except Exception as e:
            print(f"\nLỗi khi sao chép file '{img_relative_path_str}': {e}")
            error_count += 1

    # --- 5. Hoàn thành ---
    print("\n--- Hoàn thành sao chép ---")
    print(f"Đã sao chép thành công: {copied_count} ảnh.")
    print(f"Số lỗi gặp phải: {error_count}.")
    print(f"File annotation đã được lưu tại: '{DESTINATION_ANNOTATION_FILE.resolve()}'")
    print(f"Các file ảnh đã được sao chép vào thư mục: '{DESTINATION_IMAGE_DIR.resolve()}'")

if __name__ == "__main__":
    copy_data()