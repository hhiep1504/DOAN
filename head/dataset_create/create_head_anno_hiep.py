import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
from head_detection import HeadDetector
import cv2

def create_head_annotations_hiep():
    # Đường dẫn đến thư mục chứa annotations gốc
    source_ann_dir = Path("data/processed/hiep_dataset/annotations")
    # Đường dẫn đến thư mục chứa ảnh head
    head_img_dir = Path("data/processed/hiep_dataset/head_images")
    # Đường dẫn đến thư mục output cho annotations mới
    output_dir = Path("data/processed/hiep_dataset/head_annotations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo detector
    detector = HeadDetector("yolov10n-face.pt")

    # Đọc tất cả các file annotation gốc
    all_data = []
    total_annotations = 0
    matched_annotations = 0
    
    # Lấy danh sách các file annotation
    ann_files = list(source_ann_dir.glob("*.txt"))
    print(f"Found {len(ann_files)} annotation files")
    
    # Xử lý từng file annotation
    for ann_file in tqdm(ann_files, desc="Processing annotation files"):
        df = pd.read_csv(ann_file, header=None, names=[
            "frame", "id", "x", "y", "h", "w", "flag", "yaw", "pitch", "roll",
            "gender", "age", "height", "body_volume", "ethnicity", "hair_color",
            "hairstyle", "beard", "moustache", "glasses", "head_accessories",
            "upper_body_clothing", "lower_body_clothing", "feet", "accessories", "action"
        ])
        
        total_annotations += len(df)
        date_dir = ann_file.stem # Tên video, ví dụ: MVI_9421
        
        # Xử lý từng dòng trong file annotation
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {date_dir}", leave=False):
            # Tìm file ảnh head tương ứng
            person_id = int(row['id'])
            frame_num = int(row['frame'])
            
            # Tạo đường dẫn đến thư mục chứa ảnh head của person và frame này
            person_head_img_dir = head_img_dir / date_dir / 'bbox_head' / f'person_{person_id}'
            
            # Tạo pattern tìm kiếm phù hợp với format thực tế (.png)
            pattern = f"person_{person_id}_frame_{frame_num}_head.png"
            
            # Tìm kiếm file ảnh
            found = list(person_head_img_dir.glob(pattern))
            
            if found:
                # Đọc ảnh head
                img_path = found[0]
                img = cv2.imread(str(img_path))
                
                if img is not None:
                    # Detect head bbox
                    result = detector.detect_head_from_person_crop(img, conf_threshold=0.25)
                    
                    if result['success']:
                        # Lấy tọa độ bbox head
                        hx1, hy1, hx2, hy2 = result['head_bbox']
                        
                        # Lấy tên file ảnh
                        image_name = img_path.name
                        
                        # Đường dẫn tương đối từ head_images
                        relative_image_path = f"{date_dir}/bbox_head/person_{person_id}/{image_name}"
                        
                        all_data.append({
                            'imagejpg': relative_image_path,
                            'beard': int(row['beard']),
                            'glasses': int(row['glasses']),
                            'head_x1': int(hx1),
                            'head_y1': int(hy1),
                            'head_x2': int(hx2),
                            'head_y2': int(hy2)
                        })
                        matched_annotations += 1
                    else:
                        print(f"Không detect được head cho ảnh: {img_path}")
                else:
                    print(f"Không đọc được ảnh: {img_path}")

    # Tạo DataFrame và lưu file
    if all_data:
        df_new = pd.DataFrame(all_data)
        output_file = output_dir / "head_annotations.csv"
        df_new.to_csv(output_file, index=False)
        print(f"\nCreated head annotations file at: {output_file}")
        print(f"Total annotations in original files: {total_annotations}")
        print(f"Successfully matched annotations: {matched_annotations}")
        print(f"Match rate: {(matched_annotations/total_annotations)*100:.2f}%")
    else:
        print("No data was processed!")

if __name__ == "__main__":
    create_head_annotations_hiep()
