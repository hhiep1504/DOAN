# Hướng dẫn annotation lại và huấn luyện mô hình phân loại tuổi

## Mục tiêu
Annotation lại dữ liệu tuổi theo 3 nhóm chính để tăng độ chính xác của mô hình:
- Nhóm 0: 0-17 tuổi (Trẻ em)
- Nhóm 1: 18-54 tuổi (Người trưởng thành)
- Nhóm 2: 55+ tuổi (Người cao tuổi)
- Nhóm 3: Không xác định

## Các bước thực hiện

### 1. Annotation lại dữ liệu

Chạy script `reannotate_age_groups.py` để chuyển đổi các file annotation:

```bash
cd scripts/data_preparation
python reannotate_age_groups.py
```

Script này sẽ:
- Tạo backup dữ liệu gốc trong thư mục `data/raw/P-DESTRE/annotation/backup_original`
- Cập nhật tất cả các file annotation, thay đổi giá trị tuổi theo mapping mới
- In ra thống kê về số lượng bản ghi đã được thay đổi

### 2. Chuẩn bị dữ liệu huấn luyện

Sau khi annotation lại, cần tạo lại dữ liệu huấn luyện:

```bash
cd DATN
python prepare_attribute_data.py
```

Lưu ý rằng các định nghĩa nhóm tuổi trong file này đã được cập nhật để phù hợp với 3 nhóm mới.

### 3. Huấn luyện lại mô hình

Mô hình đã được cập nhật để sử dụng 4 lớp đầu ra cho thuộc tính tuổi (3 nhóm tuổi + Unknown):

```bash
cd DATN
python train_attribute_classifier.py
```

Quá trình huấn luyện sẽ sử dụng dữ liệu đã được annotation lại và lưu mô hình mới vào thư mục `attribute_data/models`.

### 4. Sử dụng mô hình mới để dự đoán

Sau khi huấn luyện xong, có thể sử dụng mô hình mới với các script dự đoán:

```bash
# Sử dụng script detect_and_classify.py
cd scripts/infer
python detect_and_classify.py --video path/to/video.mp4 --feature_model path/to/model.pth

# Hoặc sử dụng inference_yolov10.py
cd DATN
python inference_yolov10.py
```

## Các file đã được cập nhật

1. **Mô hình**: 
   - `scripts/infer/model.py`: Cập nhật đầu ra của age_classifier thành 4 lớp
   
2. **Định nghĩa dữ liệu**:
   - `scripts/data_preparation/dataset_statistic.py`: Cập nhật AGE_NAMES
   - `scripts/infer/detect_and_classify.py`: Cập nhật AGE_NAMES
   - `DATN/prepare_attribute_data.py`: Cập nhật ATTRIBUTE_MAPPINGS
   - `DATN/inference_yolov10.py`: Cập nhật ATTRIBUTE_LABELS
   - `DATN/train_attribute_classifier.py`: Cập nhật định nghĩa nhóm tuổi mới

3. **Script annotation lại dữ liệu**:
   - `scripts/data_preparation/reannotate_age_groups.py`: Script mới tạo ra

## Lưu ý quan trọng

1. **Sao lưu dữ liệu**: Script annotation tự động tạo backup, nhưng bạn nên sao lưu thêm dữ liệu gốc để đề phòng.

2. **Kiểm tra trước khi áp dụng**: Nên chạy script với một tập nhỏ file annotation trước để kiểm tra kết quả.

3. **Huấn luyện lại mô hình**: Sau khi annotation lại, bắt buộc phải huấn luyện lại mô hình vì cấu trúc đầu ra đã thay đổi.

4. **Chia sẻ thông tin thay đổi**: Đảm bảo mọi người trong nhóm đều biết về thay đổi này để tránh nhầm lẫn khi sử dụng mô hình. 