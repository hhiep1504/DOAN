import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from head_detection import HeadDetector
import cv2
import pandas as pd
import numpy as np

def extract_head_and_body_from_image(
    img_path,
    head_output_base_dir,
    body_output_base_dir,
    detector, # Nhận detector đã khởi tạo
    conf_threshold=0.25,
    base_input_dir=None # Thêm base_input_dir để tính relative path chính xác
):
    """
    Cắt head bbox và tạo body bbox từ ảnh đã cho, lưu ảnh và trả về bbox data.
    Args:
        img_path: Đường dẫn đến ảnh người đã crop (fullbody)
        head_output_base_dir: Thư mục gốc lưu ảnh head
        body_output_base_dir: Thư mục gốc lưu ảnh body
        detector: Instance của HeadDetector đã khởi tạo
        conf_threshold: Ngưỡng confidence cho face detection
        base_input_dir: Thư mục gốc chứa ảnh input (để tính relative path)
    Returns:
        Tuple chứa (head_bbox_data_list, body_bbox_data_list) cho ảnh này, hoặc ([], []) nếu lỗi.
    """
    img_path = Path(img_path)
    head_output_base_dir = Path(head_output_base_dir)
    body_output_base_dir = Path(body_output_base_dir)

    person_crop = cv2.imread(str(img_path))
    if person_crop is None:
        print(f"Không load được ảnh: {img_path}")
        return [], []

    result = detector.detect_head_from_person_crop(person_crop, conf_threshold=conf_threshold)

    head_bbox_data = []
    body_bbox_data = []

    if result['success']:
        hx1, hy1, hx2, hy2 = result['head_bbox']
        head_crop = person_crop[hy1:hy2, hx1:hx2]

        if head_crop.size == 0:
            print(f"Head crop rỗng: {img_path}")
            return [], []

        # Tạo đường dẫn tương đối từ thư mục gốc input
        if base_input_dir:
             try:
                 # Lấy phần đường dẫn từ base_input_dir đến thư mục chứa ảnh hiện tại
                 relative_segment = img_path.relative_to(Path(base_input_dir)).parent
             except ValueError:
                 print(f"Could not get relative path for {img_path} relative to {base_input_dir}")
                 relative_segment = Path("") # Fallback to empty if cannot determine relative path
        else:
            # Nếu không có base_input_dir, dùng relative path từ thư mục ảnh input
            relative_segment = img_path.parent.relative_to(img_path.parent.parent) # Có thể cần chỉnh lại logic này tùy cấu trúc

        # Tạo đường dẫn output đầy đủ
        head_output_dir = head_output_base_dir / relative_segment
        head_output_dir.mkdir(parents=True, exist_ok=True)
        head_output_path = head_output_dir / f"{img_path.stem}_head{img_path.suffix}"

        body_output_dir = body_output_base_dir / relative_segment
        body_output_dir.mkdir(parents=True, exist_ok=True)
        body_output_path = body_output_dir / f"{img_path.stem}_body{img_path.suffix}"

        # Lưu ảnh head crop
        cv2.imwrite(str(head_output_path), head_crop)

        # Tạo và lưu ảnh body crop (cắt phần dưới head)
        body_crop = person_crop[hy2:, :]
        cv2.imwrite(str(body_output_path), body_crop)

        # Lưu thông tin bbox head
        head_bbox_data.append({
            'image_path': str(head_output_path.relative_to(head_output_base_dir)),
            'x1': int(hx1),
            'y1': int(hy1),
            'x2': int(hx2),
            'y2': int(hy2),
            'confidence': float(result['confidence'])
        })

        # Lưu thông tin bbox body
        # Bbox trong ảnh body đã crop sẽ bắt đầu từ (0,0) và kết thúc ở kích thước ảnh body
        body_bbox_data.append({
            'image_path': str(body_output_path.relative_to(body_output_base_dir)),
            'x1': 0,
            'y1': 0,
            'x2': body_crop.shape[1],
            'y2': body_crop.shape[0]
        })
    else:
        print(f"❌ Không detect được head cho {img_path}: {result['error']}")

    return head_bbox_data, body_bbox_data

def process_pdestre_dataset(base_dir, head_output_base_dir, body_output_base_dir, batch_size=100):
    """
    Xử lý toàn bộ dataset P-DESTRE, trích xuất head/body và tạo annotations
    Args:
        base_dir: Thư mục gốc chứa ảnh đã crop fullbody
        head_output_base_dir: Thư mục gốc đầu ra cho ảnh head (vd: OUTPUT_DIR/head)
        body_output_base_dir: Thư mục gốc đầu ra cho ảnh body (vd: data/body)
        batch_size: Số lượng ảnh xử lý trong một batch
    """
    base_dir = Path(base_dir)
    head_output_base_dir = Path(head_output_base_dir)
    body_output_base_dir = Path(body_output_base_dir)

    head_output_base_dir.mkdir(parents=True, exist_ok=True)
    body_output_base_dir.mkdir(parents=True, exist_ok=True)

    all_head_bbox_data = []
    all_body_bbox_data = []

    # Lấy danh sách tất cả các file ảnh .jpg trong base_dir
    all_image_paths = list(base_dir.glob("**/*.jpg"))
    print(f"Tìm thấy {len(all_image_paths)} ảnh trong dataset P-DESTRE")

    # Khởi tạo detector ở đây để tái sử dụng
    detector = HeadDetector("yolov10n-face.pt") # Thay bằng đường dẫn model thật nếu cần

    # Xử lý ảnh theo batch
    for i in tqdm(range(0, len(all_image_paths), batch_size), desc="Processing batches"):
        batch_paths = all_image_paths[i:i+batch_size]
        # print(f"Xử lý batch {i//batch_size + 1}/{(len(all_image_paths)-1)//batch_size + 1}")

        # Xử lý từng ảnh trong batch
        for img_path in tqdm(batch_paths, desc=f"Processing images in batch {i//batch_size + 1}", leave=False):
            head_data, body_data = extract_head_and_body_from_image(
                img_path,
                head_output_base_dir,
                body_output_base_dir,
                detector,
                conf_threshold=0.25,
                base_input_dir=base_dir # Truyền base input dir
            )
            all_head_bbox_data.extend(head_data)
            all_body_bbox_data.extend(body_data)

        # Giải phóng bộ nhớ
        import gc
        gc.collect()

    # Lưu tất cả thông tin vào 2 file CSV duy nhất
    # Lưu CSV Head tại thư mục cha của head_output_base_dir
    if all_head_bbox_data:
        head_df = pd.DataFrame(all_head_bbox_data)
        head_csv_path = head_output_base_dir.parent / 'head_annotations.csv'
        head_df.to_csv(head_csv_path, index=False)
        print(f"\nĐã lưu thông tin bbox head dataset P-DESTRE vào: {head_csv_path}")

    # Lưu CSV Body tại thư mục gốc body_output_base_dir
    if all_body_bbox_data:
        body_df = pd.DataFrame(all_body_bbox_data)
        body_csv_path = body_output_base_dir / 'body_annotations.csv'
        body_df.to_csv(body_csv_path, index=False)
        print(f"Đã lưu thông tin bbox body dataset P-DESTRE vào: {body_csv_path}")

    print("\nHoàn tất xử lý dataset P-DESTRE.")

def process_hiep_dataset(base_dir, head_output_base_dir, body_output_base_dir, batch_size=100):
    """
    Xử lý dataset Hiep, trích xuất head/body và tạo annotations
    Args:
        base_dir: Thư mục gốc chứa ảnh đã crop fullbody (hiep_dataset/images)
        head_output_base_dir: Thư mục gốc đầu ra cho ảnh head (vd: HIEP_OUTPUT_DIR/head)
        body_output_base_dir: Thư mục gốc đầu ra cho ảnh body (vd: data/body)
        batch_size: Số lượng ảnh xử lý trong một batch
    """
    base_dir = Path(base_dir)
    head_output_base_dir = Path(head_output_base_dir)
    body_output_base_dir = Path(body_output_base_dir)

    head_output_base_dir.mkdir(parents=True, exist_ok=True)
    body_output_base_dir.mkdir(parents=True, exist_ok=True)

    all_head_bbox_data = []
    all_body_bbox_data = []

    video_dirs = [d for d in base_dir.iterdir() if d.is_dir()]

    # Khởi tạo detector ở đây để tái sử dụng
    detector = HeadDetector("yolov10n-face.pt") # Thay bằng đường dẫn model thật nếu cần

    for video_dir in tqdm(video_dirs, desc="Xử lý video"):
        video_name = video_dir.name

        bbox_dir = video_dir / "bbox_image"
        if not bbox_dir.exists():
            print(f"Không tìm thấy thư mục bbox_image trong {video_name}")
            continue

        person_dirs = [d for d in bbox_dir.iterdir() if d.is_dir()]

        for person_dir in tqdm(person_dirs, desc=f"Xử lý {video_name}", leave=False):
            person_id = person_dir.name

            # Lấy tất cả hình ảnh của person
            image_paths = list(person_dir.glob("*.png"))

            # Xử lý từng ảnh để detect và cắt ảnh
            for img_path in tqdm(image_paths, desc=f"Processing images in {person_id}", leave=False):
                head_data, body_data = extract_head_and_body_from_image(
                    img_path,
                    head_output_base_dir,
                    body_output_base_dir,
                    detector,
                    conf_threshold=0.25,
                    base_input_dir=base_dir # Truyền base input dir
                )
                all_head_bbox_data.extend(head_data)
                all_body_bbox_data.extend(body_data)

        # Giải phóng bộ nhớ
        import gc
        gc.collect()

    # Lưu tất cả thông tin vào 2 file CSV duy nhất tại HIEP_OUTPUT_DIR
    if all_head_bbox_data:
        head_df = pd.DataFrame(all_head_bbox_data)
        head_csv_path = head_output_base_dir.parent / 'head_annotations.csv'
        head_df.to_csv(head_csv_path, index=False)
        print(f"\nĐã lưu thông tin bbox head dataset Hiep vào: {head_csv_path}")

    if all_body_bbox_data:
        body_df = pd.DataFrame(all_body_bbox_data)
        body_csv_path = body_output_base_dir / 'body_annotations.csv'
        body_df.to_csv(body_csv_path, index=False)
        print(f"Đã lưu thông tin bbox body dataset Hiep vào: {body_csv_path}")

    print("\nHoàn tất xử lý dataset Hiep.")

if __name__ == "__main__":
    # Thay đổi đường dẫn này theo cấu trúc thư mục của bạn
    BASE_DATA_PROCESSED = "data/processed"

    PDESTRE_INPUT_DIR = Path(BASE_DATA_PROCESSED) / "pdestre" / "images"
    PDESTRE_OUTPUT_BASE_DIR = Path("data/body") / "pdestre"  # Thư mục chung cho output
    PDESTRE_HEAD_OUTPUT_DIR = PDESTRE_OUTPUT_BASE_DIR / "head_images"  # Ảnh head lưu tại đây
    PDESTRE_BODY_OUTPUT_DIR = Path("data/body") / "pdestre" / "body_images" # Ảnh body P-DESTRE lưu tại đây

    HIEP_INPUT_DIR = Path(BASE_DATA_PROCESSED) / "hiep_dataset" / "images"
    HIEP_OUTPUT_BASE_DIR = Path(BASE_DATA_PROCESSED) / "hiep_dataset" / "processed" # Thư mục chung cho output
    HIEP_HEAD_OUTPUT_DIR = HIEP_OUTPUT_BASE_DIR / "head_images" / "head" # Ảnh head lưu tại đây
    HIEP_BODY_OUTPUT_DIR = Path("data/body") / "hiep_dataset" / "images" # Ảnh body Hiep lưu tại đây

    # Chọn dataset cần xử lý (uncomment dòng tương ứng)

    # Xử lý P-DESTRE dataset
    process_pdestre_dataset(PDESTRE_INPUT_DIR, PDESTRE_HEAD_OUTPUT_DIR, PDESTRE_BODY_OUTPUT_DIR, batch_size=100)

    # Xử lý Hiep dataset
    # process_hiep_dataset(HIEP_INPUT_DIR, HIEP_HEAD_OUTPUT_DIR, HIEP_BODY_OUTPUT_DIR, batch_size=100)
