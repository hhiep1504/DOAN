import json
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd # Import pandas for easier data handling if needed, although direct list processing is fine here

def convert_hiep_annotations_to_pdestre_format():
    # Đường dẫn đến thư mục hiep_dataset
    hiep_dir = Path("outputs/hiep_dataset") # Giả định thư mục hiep_dataset ở gốc workspace
    output_dir = Path("data/processed/hiep_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Duyệt qua từng video trong hiep_dataset
    for video_dir in hiep_dir.iterdir():
        if not video_dir.is_dir():
            continue
            
        print(f"Đang xử lý video: {video_dir.name}")
        
        # Đọc file annotations.json
        annotation_file_path = video_dir / "annotations.json"
        if not annotation_file_path.exists():
            print(f"Không tìm thấy file annotations.json cho {video_dir.name}")
            continue
            
        try:
            with open(annotation_file_path, 'r') as f:
                annotations_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Lỗi giải mã JSON trong file {annotation_file_path}")
            continue
        except Exception as e:
            print(f"Lỗi khi đọc file {annotation_file_path}: {e}")
            continue
            
        # Tạo file annotation mới theo định dạng P-DESTRE
        output_file = output_dir / f"{video_dir.name}.txt"
        
        with open(output_file, 'w') as f:
            # Các giá trị mặc định/placeholder cho các cột không có trong annotations.json
            # Dựa trên phân tích định dạng P-DESTRE trước đó
            # frame, person_id, x, y, w, h, ?, ?, ?, ?, gender, age, ?, ethnicity, ?, ?, beard, ?, glasses, ?, ?, accessories, ?, ?
            # Chỉ số 0 -> 25
            
            # Sắp xếp frame_str để đảm bảo thứ tự frame tăng dần
            sorted_frames = sorted(annotations_data.keys(), key=lambda x: int(x))
            
            for frame_str in sorted_frames:
                frame_num = int(frame_str)
                detections = annotations_data[frame_str]
                
                for det in detections:
                    try:
                        person_id = det['track_id']
                        bbox_xyxy = det['bbox']
                        attrs = det['attributes']
                        
                        # Chuyển đổi bbox từ [x1, y1, x2, y2] sang [x, y, w, h]
                        x1, y1, x2, y2 = bbox_xyxy
                        x, y = x1, y1
                        w, h = x2 - x1, y2 - y1
                        
                        # Làm tròn các giá trị bounding box đến 1 chữ số thập phân
                        x_rounded = round(x, 1)
                        y_rounded = round(y, 1)
                        w_rounded = round(w, 1)
                        h_rounded = round(h, 1)
                        
                        # Chuẩn bị các cột cho định dạng P-DESTRE (26 cột)
                        # Khởi tạo với các placeholder
                        pdestre_row = [0] * 26
                        
                        pdestre_row[0] = frame_num      # frame
                        pdestre_row[1] = person_id      # person_id
                        pdestre_row[2] = x_rounded      # x
                        pdestre_row[3] = y_rounded      # y
                        pdestre_row[4] = w_rounded      # w
                        pdestre_row[5] = h_rounded      # h
                        
                        # Điền các thuộc tính từ annotations.json vào đúng vị trí
                        # Cột 11 -> index 10: gender
                        pdestre_row[10] = attrs.get('gender', 2) # Mặc định 2 nếu không có (Unknown)
                        # Cột 12 -> index 11: age
                        pdestre_row[11] = attrs.get('age', 8) # Mặc định 8 nếu không có (Unknown)
                        # Cột 15 -> index 14: ethnicity
                        pdestre_row[14] = attrs.get('ethnicity', 4) # Mặc định 4 nếu không có (Unknown)
                        # Cột 18 -> index 17: beard
                        pdestre_row[17] = attrs.get('beard', 2) # Mặc định 2 nếu không có (Không rõ)
                         # Cột 20 -> index 19: glasses
                        pdestre_row[19] = attrs.get('glasses', 3) # Mặc định 3 nếu không có (Không rõ)
                        # Cột 25 -> index 24: accessories
                        pdestre_row[24] = attrs.get('accessories', 7) # Mặc định 7 nếu không có (Không có)
                        
                        # Chuyển đổi list sang chuỗi ngăn cách bởi dấu phẩy
                        # Đảm bảo các giá trị float được định dạng đúng với 1 chữ số thập phân
                        line = ','.join(f"{val:.1f}" if isinstance(val, float) else str(val) for val in pdestre_row)
                        f.write(line + '\n')
                        
                    except Exception as e:
                        print(f"Lỗi khi xử lý detection trong frame {frame_str}: {e}")
                        continue
                        
        print(f"Đã tạo file annotation: {output_file}")

if __name__ == "__main__":
    convert_hiep_annotations_to_pdestre_format()
