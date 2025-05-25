"""
 Lọc danh sách ảnh dựa trên việc phát hiện đối tượng 'person' bằng mô hình YOLO.
Chức năng:
- Đọc danh sách các đường dẫn ảnh "ứng viên" từ file text.
- Xác minh sự tồn tại của các file ảnh song song để tăng tốc.
- Sử dụng mô hình YOLO (ví dụ: YOLOv8) để phát hiện người trong các ảnh theo lô (batch processing).
- Áp dụng các bộ lọc dựa trên:
    - Ngưỡng tin cậy (confidence score) của phát hiện.
    - Tỷ lệ diện tích tối thiểu của bounding box người so với diện tích ảnh.
    - Chỉ giữ lại ảnh có chứa ít nhất một người thỏa mãn các tiêu chí.
- Tối ưu hóa hiệu suất bằng cách sử dụng GPU (nếu có), xử lý theo lô, và tính toán nửa độ chính xác (FP16).
- Lưu danh sách đường dẫn của các ảnh đã được lọc vào một file text mới.
- Mục tiêu là loại bỏ các ảnh không chứa người hoặc người quá nhỏ/không rõ ràng.
"""
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm
import math
import concurrent.futures
import os
import numpy as np

# Thêm hàm trợ giúp để chia list thành các batch nhỏ
def chunkify(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Thêm hàm verify_file để kiểm tra nhanh các tệp tin
def verify_file(file_path):
    """Kiểm tra xem tệp tin có tồn tại và có thể đọc được không"""
    path = Path(file_path)
    return path.exists() and path.is_file()

# Chức năng để tiền xử lý danh sách đường dẫn song song
def preprocess_paths(candidate_file, root_path, max_workers=None):
    """Đọc và xác minh danh sách đường dẫn song song để tăng tốc"""
    candidate_full_paths = []
    candidate_relative_paths_map = {}
    
    try:
        # Đọc tất cả các đường dẫn từ file
        with open(candidate_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        # Chuẩn bị danh sách đường dẫn đầy đủ
        relative_paths = [line.replace('\\', '/') for line in lines]
        full_paths = [root_path / rel_path for rel_path in relative_paths]
        
        # Kiểm tra song song các đường dẫn
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(verify_file, full_paths))
        
        # Xây dựng mapping và danh sách cuối cùng
        for i, (full_path, exists) in enumerate(zip(full_paths, results)):
            if exists:
                candidate_full_paths.append(full_path)
                candidate_relative_paths_map[str(full_path)] = relative_paths[i]
        
        return candidate_full_paths, candidate_relative_paths_map
    
    except Exception as e:
        print(f"Error in parallel path processing: {e}")
        return [], {}

def filter_images_by_content_batched(
    candidate_paths_file,
    output_file,
    jpg_root_dir,
    yolo_model_path,
    device,
    batch_size=32,
    num_workers=4,
    conf_threshold=0.5,
    min_area_ratio=0.1,
    person_class_id=0,
    half_precision=True,  # Sử dụng FP16 để tăng tốc khi có GPU
    preprocess_workers=None  # Số workers cho tiền xử lý đường dẫn
):
    """
    Filters images based on person detection using YOLO with batch processing.

    Args:
        candidate_paths_file (str or Path): Path to the text file containing candidate image paths (relative).
        output_file (str or Path): Path to save the final list of kept image paths.
        jpg_root_dir (str or Path): Path to the root directory of JPGs.
        yolo_model_path (str or Path): Path to the YOLO model file.
        device (str): Device to run YOLO on ('cuda' or 'cpu').
        batch_size (int): Number of images to process in each YOLO batch.
        num_workers (int): Number of worker processes for data loading.
        conf_threshold (float): Minimum confidence score for person detection.
        min_area_ratio (float): Minimum ratio of person bounding box area to image area.
        person_class_id (int): The class ID corresponding to 'person'.
        half_precision (bool): Whether to use FP16 precision when device is cuda
        preprocess_workers (int): Number of workers for path preprocessing
    """
    candidate_file = Path(candidate_paths_file)
    output_path = Path(output_file)
    root_path = Path(jpg_root_dir)

    if not candidate_file.exists():
        print(f"Error: Candidate paths file not found at {candidate_file}")
        return

    # --- Load YOLO Model ---
    print(f"Loading YOLO model from {yolo_model_path} onto {device}...")
    try:
        # Thêm xử lý lỗi UnpicklingError
        try:
            import ultralytics.nn.tasks
            if hasattr(ultralytics.nn.tasks, 'YOLOv10DetectionModel'):
                 torch.serialization.add_safe_globals([ultralytics.nn.tasks.YOLOv10DetectionModel])
            elif hasattr(ultralytics.nn.tasks, 'DetectionModel'):
                 torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
        except (ImportError, AttributeError) as e:
            print(f"Warning: Could not setup safe globals for model loading: {e}. Proceeding without.")

        model = YOLO(yolo_model_path)
        print("YOLO model loaded successfully.")

        # Optimize model for inference
        if half_precision and device == 'cuda':
            print("Using half precision (FP16) for faster inference")
        
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # --- Read Candidate Paths (parallel processing) ---
    print(f"Reading candidate paths from {candidate_file} with parallel processing...")
    if preprocess_workers is None:
        preprocess_workers = os.cpu_count()
    
    start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    
    if start_time:
        start_time.record()
    
    candidate_full_paths, candidate_relative_paths_map = preprocess_paths(
        candidate_file, root_path, max_workers=preprocess_workers
    )
    
    print(f"Read and verified {len(candidate_full_paths)} existing candidate image paths.")

    if not candidate_full_paths:
         print("No valid candidate paths found.")
         return

    # --- Filter Images using Batched Inference ---
    final_kept_relative_paths = []
    processed_count = 0
    error_count = 0

    # Tạo các lô đường dẫn ảnh đầy đủ
    path_batches = list(chunkify(candidate_full_paths, batch_size))
    print(f"Processing {len(candidate_full_paths)} images in {len(path_batches)} batches of size up to {batch_size}...")
    print(f"Using {num_workers} workers for data loading...")

    # Tối ưu cho quy trình xử lý batch
    for batch_paths in tqdm(path_batches, desc="Filtering images by content (batched)"):
        batch_relative_paths = [candidate_relative_paths_map[str(p)] for p in batch_paths]
        try:
            # Chạy dự đoán YOLO trên cả lô với các tùy chọn tối ưu hóa
            results = model(
                batch_paths, 
                verbose=False, 
                device=device, 
                conf=conf_threshold, 
                stream=False,
                workers=num_workers,
                half=half_precision and device == 'cuda'  # Sử dụng FP16 nếu có GPU
            )
            processed_count += len(batch_paths)

            # Xử lý kết quả và áp dụng điều kiện lọc
            kept_indices = []
            for i, result in enumerate(results):
                # Lấy kích thước gốc từ kết quả YOLO
                img_height, img_width = result.orig_shape
                if img_width == 0 or img_height == 0: 
                    continue # Bỏ qua nếu kích thước không hợp lệ
                
                img_area = img_width * img_height
                boxes = result.boxes

                # Chỉ xem xét các box có lớp và confidence phù hợp
                person_boxes = []
                for box in boxes:
                    if int(box.cls.item()) == person_class_id and box.conf.item() >= conf_threshold:
                        person_boxes.append(box)
                
                if not person_boxes:
                    continue  # Không có người trong ảnh, bỏ qua
                
                # Kiểm tra tỷ lệ diện tích
                for box in person_boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox_area = (x2 - x1) * (y2 - y1)
                    area_ratio = bbox_area / img_area if img_area > 0 else 0
                    
                    if area_ratio >= min_area_ratio:
                        kept_indices.append(i)
                        break

            # Thu thập đường dẫn của các ảnh được giữ lại
            final_kept_relative_paths.extend([batch_relative_paths[i] for i in kept_indices])

        except Exception as e:
            print(f"\nError processing batch starting with {batch_paths[0].name}: {e}. Skipping batch.")
            error_count += len(batch_paths)
            continue

    if end_time:
        end_time.record()
        torch.cuda.synchronize()
        processing_time = start_time.elapsed_time(end_time) / 1000.0  # ms to seconds
        print(f"\nTotal processing time: {processing_time:.2f} seconds")

    # --- Save Results ---
    print("\n--- Content Filtering Summary ---")
    total_candidates_read = len(candidate_relative_paths_map) # Tổng số đường dẫn ban đầu có file tồn tại
    print(f"Total existing candidate images read: {total_candidates_read}")
    print(f"Images processed by YOLO: {processed_count}")
    print(f"Images skipped due to processing errors: {error_count}")
    print(f"Images kept after content filtering: {len(final_kept_relative_paths)}")
    removed_count = processed_count - error_count - len(final_kept_relative_paths)
    print(f"Images removed due to quality/content criteria: {removed_count}")

    print(f"\nSaving {len(final_kept_relative_paths)} paths to: {output_path.resolve()}...")
    try:
        with open(output_path, 'w') as f:
            for path_str in final_kept_relative_paths:
                f.write(f"{path_str}\n")
        print("Save complete.")
    except Exception as e:
        print(f"Error saving output file: {e}")


# --- Cấu hình ---
JPG_ROOT_DIR = 'data/raw/jpg_Extracted_PIDS'  # Đường dẫn gốc chứa các thư mục ngày
CANDIDATE_PATHS_FILE = 'final_content_filtered.txt'  # Input: File từ bước lọc trùng lặp
OUTPUT_FILE = 'final.txt'  # Output: Đổi tên để tránh ghi đè
YOLO_MODEL_PATH = 'yolov8n.pt'  # Đường dẫn đến file model YOLO của bạn
BATCH_SIZE = 18  # Kích thước lô (TĂNG LÊN, điều chỉnh dựa trên VRAM GPU)
CONF_THRESHOLD = 0.7  # Ngưỡng tin cậy
MIN_AREA_RATIO = 0.2  # Tỷ lệ diện tích tối thiểu
PERSON_CLASS_ID = 0  # ID của lớp 'person'
NUM_WORKERS = 4  # Số lượng worker processes cho data loading
USE_HALF_PRECISION = True  # Sử dụng FP16 để tăng tốc khi có GPU

# --- Chạy quá trình lọc ---
if __name__ == "__main__":
    # Xác định thiết bị
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: {device} ({gpu_name})")
        # In thông tin bộ nhớ để ước tính batch size phù hợp
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total GPU Memory: {total_mem:.2f} GB")
        
        # Tự động điều chỉnh batch size dựa trên dung lượng GPU
        recommended_batch_size = min(32, int(total_mem * 2))  # Heuristic: 2 batch/GB
        if recommended_batch_size > BATCH_SIZE:
            choice = input(f"Detected {total_mem:.1f}GB GPU memory. Increase batch size from {BATCH_SIZE} to {recommended_batch_size}? (y/n): ")
            if choice.lower() == 'y':
                BATCH_SIZE = recommended_batch_size
                print(f"Using increased batch size: {BATCH_SIZE}")
    else:
        device = 'cpu'
        print(f"Using device: {device}")
        # CPU không được hưởng lợi từ half precision
        USE_HALF_PRECISION = False

    # Tự động điều chỉnh num_workers
    import os
    recommended_workers = max(1, os.cpu_count() - 1)  # Để lại 1 core cho hệ thống
    if recommended_workers != NUM_WORKERS:
        print(f"Recommended workers for your system: {recommended_workers}")
        choice = input(f"Use recommended {recommended_workers} workers instead of {NUM_WORKERS}? (y/n): ")
        if choice.lower() == 'y':
            NUM_WORKERS = recommended_workers
            print(f"Using {NUM_WORKERS} workers")

    # Số workers cho tiền xử lý đường dẫn
    PREPROCESS_WORKERS = os.cpu_count()

    # Chạy hàm lọc với các tối ưu hóa
    filter_images_by_content_batched(
        CANDIDATE_PATHS_FILE,
        OUTPUT_FILE,
        JPG_ROOT_DIR,
        YOLO_MODEL_PATH,
        device,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        conf_threshold=CONF_THRESHOLD,
        min_area_ratio=MIN_AREA_RATIO,
        person_class_id=PERSON_CLASS_ID,
        half_precision=USE_HALF_PRECISION,
        preprocess_workers=PREPROCESS_WORKERS
    )