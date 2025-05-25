

## Cài đặt

1.  **Clone repository (Nếu có):**
    ```bash
    git clone <your-repository-url>
    cd <your-project-root>
    ```

2.  **Tạo môi trường ảo (Khuyến nghị):**
    ```bash
    python -m venv venv
    # Trên Windows
    .\venv\Scripts\activate
    # Trên macOS/Linux
    source venv/bin/activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    Tạo file `requirements.txt` với nội dung sau:
    ```txt
    torch
    torchvision
    Pillow
    matplotlib
    numpy
    pandas
    tqdm
    opencv-python
    ultralytics
    ```
    Sau đó chạy lệnh:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Tải các model cần thiết:**
    *   **YOLOv10:** Tải file `yolov10n.pt` (hoặc phiên bản khác nếu bạn thay đổi trong code) từ repository của YOLOv10 và đặt vào thư mục gốc của project.
    *   **Feature Classifier:** Đảm bảo bạn có file `best_model.pth` (model đã được huấn luyện) trong thư mục gốc. Nếu chưa có, bạn cần chạy script huấn luyện (xem phần Huấn luyện).

5.  **Chuẩn bị dữ liệu (Nếu cần huấn luyện hoặc kiểm tra dataset):**
    *   Tải dataset P-DESTRE.
    *   Giải nén và đặt thư mục `P-DESTRE/annotation` vào thư mục gốc.
    *   Đảm bảo thư mục `jpg_Extracted_PIDS` chứa các ảnh đã được trích xuất và sắp xếp theo cấu trúc như mô tả ở trên. Script này giả định ảnh đã được trích xuất sẵn.

## Huấn luyện Model (Tùy chọn)

Script `train.py` dùng để huấn luyện model `FeatureClassifier` dựa trên một file annotation cụ thể từ P-DESTRE.

1.  **Cấu hình:** Mở file `train.py` và chỉnh sửa các đường dẫn (`jpg_dir`, `annotation_file`) nếu cần. Hiện tại đang dùng `P-DESTRE/annotation/08-11-2019-1-1.txt`. Bạn có thể sửa để dùng `ConcatDataset` nếu muốn huấn luyện trên toàn bộ dữ liệu.
2.  **Chạy huấn luyện:**
    ```bash
    python train.py
    ```
    Quá trình huấn luyện sẽ diễn ra, và model tốt nhất (dựa trên validation loss) sẽ được lưu vào `best_model.pth`.

## Kiểm tra và Thống kê Dataset

Script `dataset_statistic.py` giúp kiểm tra cấu trúc dữ liệu, tải các mẫu, tính toán thống kê phân phối các đặc điểm (gender, age, ethnicity) và hiển thị một số mẫu ngẫu nhiên.

1.  **Cấu hình:** Mở file `dataset_statistic.py` và đảm bảo các đường dẫn `JPG_DIR` và `ANNOTATION_DIR` là chính xác. Bạn cũng có thể thay đổi `NUM_SAMPLES_TO_CHECK` và `NUM_SAMPLES_TO_SHOW`.
2.  **Chạy kiểm tra:**
    ```bash
    python dataset_statistic.py
    ```
    Script sẽ in ra các thông tin kiểm tra, thống kê và hiển thị ảnh mẫu (nếu không có lỗi).

## Sử dụng Model (Inference)

Có hai cách chính để sử dụng model đã huấn luyện:

### 1. Phân loại đặc điểm trên ảnh đơn (Giả định ảnh chỉ chứa một người và đã crop)

Script `infer.py` tải model `FeatureClassifier` và chạy dự đoán trên một file ảnh đơn. Ảnh này nên chỉ chứa khuôn mặt hoặc phần lớn cơ thể của một người.

1.  **Cấu hình:** Mở file `infer.py` và chỉnh sửa biến `test_image` thành đường dẫn đến ảnh bạn muốn thử nghiệm. Đảm bảo `model_path` trỏ đúng đến file `best_model.pth`.
2.  **Chạy inference:**
    ```bash
    python infer.py
    ```
    Script sẽ tải model, xử lý ảnh, đưa ra dự đoán về giới tính, tuổi, dân tộc và hiển thị ảnh gốc cùng kết quả dự đoán.

### 2. Phát hiện người và Phân loại đặc điểm

Script `detect_and_classify.py` sử dụng YOLOv10 để phát hiện tất cả người trong ảnh, sau đó cắt ảnh từng người và dùng `FeatureClassifier` để phân loại đặc điểm cho mỗi người.

1.  **Cấu hình:**
    *   Mở file `detect_and_classify.py`.
    *   Đảm bảo các đường dẫn đến model YOLO (`yolov10n.pt`) và `FeatureClassifier` (`best_model.pth`) trong hàm `load_models` là chính xác.
    *   Chỉnh sửa biến `test_image` thành đường dẫn đến ảnh bạn muốn xử lý.
2.  **Chạy phát hiện và phân loại:**
    ```bash
    python detect_and_classify.py
    ```
    Script sẽ:
    *   Tải cả hai model.
    *   Đọc ảnh đầu vào.
    *   Dùng YOLO để tìm bounding box của người.
    *   Với mỗi người được phát hiện:
        *   Cắt ảnh người đó ra.
        *   Dùng `FeatureClassifier` để dự đoán đặc điểm.
        *   In thông tin dự đoán ra terminal.
        *   Vẽ bounding box và thông tin dự đoán lên ảnh gốc.
    *   Hiển thị ảnh kết quả cuối cùng.
    *   Lưu ảnh kết quả vào file `result.jpg`.

## Các Script Khác

*   `model.py`: Chứa định nghĩa lớp `FeatureClassifier` sử dụng ResNet50 làm backbone.
*   `dataset_multiannotation_gemini.py`: Định nghĩa lớp `PdestreFeatureDataset` để đọc và xử lý dữ liệu từ các file annotation và thư mục ảnh của P-DESTRE. Script này được sử dụng bởi `train.py` và `dataset_statistic.py`.
*   `read_annotations.py`: Một script tiện ích để đọc và hiển thị nội dung của một file annotation, giúp hiểu ý nghĩa của các cột dữ liệu.

doan/
├── .gitignore              # Các file/thư mục bỏ qua khi dùng Git
├── README.md               # Mô tả tổng quan về dự án, cách cài đặt, sử dụng
├── requirements.txt        # Các thư viện Python cần thiết (pip install -r requirements.txt)
│
├── config/                 # (Tùy chọn) Chứa các file cấu hình (vd: YAML, JSON) cho đường dẫn, tham số model
│   └── config.yaml
│
├── data/
│   ├── raw/                # Dữ liệu gốc, không thay đổi
│   │   ├── jpg_Extracted_PIDS/
│   │   │   └── ... (Các thư mục ngày và ảnh gốc)
│   │   └── P-DESTRE/
│   │       └── annotation/
│   │           └── ... (Các file annotation gốc .txt)
│   │
│   ├── interim/            # Dữ liệu trung gian sau một số bước xử lý
│   │   ├── P-DESTRE/
│   │   │   └── annotation_cleaned/
│   │   │       └── ... (Annotation đã lọc trùng dòng)
│   │   ├── kept_image_paths.txt
│   │   └── final_kept_image_paths.txt # (Có thể đổi tên test.txt thành cái này nếu đúng mục đích)
│   │
│   └── processed/          # Dữ liệu cuối cùng sẵn sàng cho model
│       └── P-DESTRE/
│           └── annotation_final/
│               └── ... (Annotation cuối cùng đã lọc)
│       # (Có thể có thêm các file train/val/test split ở đây)
│
├── models/                 # Chứa các file trọng số model đã huấn luyện hoặc tải về
│   ├── yolov8n.pt
│   ├── yolov10n.pt
│   └── checkpoints/        # Nơi lưu các checkpoint trong quá trình huấn luyện
│       ├── best_model.pth
│       └── best_weight.pth
│
├── notebooks/              # (Tùy chọn) Chứa các Jupyter Notebook cho việc khám phá, thử nghiệm
│   └── exploratory_analysis.ipynb
│
├── outputs/                # Kết quả đầu ra từ các script (ảnh, log, báo cáo)
│   ├── images/
│   │   ├── result.jpg
│   │   ├── save1.png
│   │   ├── save2.png
│   │   └── test.webp # (?) Xem xét lại file này
│   ├── logs/               # File log quá trình huấn luyện, inference
│   └── reports/            # Các báo cáo, bảng biểu
│       └── HaHoangHiep_PGNV_DATN.xlsx
│
├── scripts/                # Các script chạy độc lập
│   ├── data_preparation/   # Script xử lý, chuẩn bị dữ liệu
│   │   ├── 01_clean_annotations.py   # (Script tạo annotation_cleaned)
│   │   ├── 02_filter_similar_images.py # Đổi tên từ filter_images.py
│   │   ├── 03_filter_quality_yolo.py   # Đổi tên từ filter_yolo.py
│   │   ├── 04_create_final_annotations.py # Đổi tên từ create_new_anno.py
│   │   └── 05_split_dataset.py        # Đổi tên từ split_dataset.py
│   │
│   ├── analysis/           # Script phân tích, thống kê dữ liệu
│   │   ├── dataset_statistics.py    # Đổi tên từ dataset_statistic.py
│   │   └── read_annotations.py      # Đổi tên từ read_annotations.py
│   │
│   ├── training/           # Script huấn luyện model
│   │   └── train.py               # Đổi tên từ train.py
│   │
│   ├── inference/          # Script chạy dự đoán, phân loại
│   │   ├── infer.py               # Đổi tên từ infer.py
│   │   └── detect_and_classify.py # Xem xét gộp với infer.py nếu logic tương tự
│   │
│   └── visualization/      # Script hiển thị kết quả, dữ liệu
│       └── show.py                # Đổi tên từ show.py
│
└── src/                    # Mã nguồn chính (các lớp, hàm tái sử dụng)
    ├── __init__.py         # Đánh dấu là Python package
    ├── datasets.py         # Định nghĩa các lớp Dataset (vd: PdestreFeatureDataset từ dataset_multiannotation_gemini.py)
    ├── models.py           # Định nghĩa kiến trúc model tùy chỉnh (vd: feature extractor từ model.py)
    └── utils.py            # Các hàm tiện ích dùng chung (vd: visualize_sample, parse_image_path...)


