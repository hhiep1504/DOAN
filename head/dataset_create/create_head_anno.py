import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm

def create_head_annotations():
    # Đường dẫn đến thư mục chứa annotations gốc
    source_ann_dir = Path("data/processed/pdestre/annotations")
    # Đường dẫn đến thư mục chứa ảnh head (gốc)
    head_img_dir = Path("data/processed/pdestre/head_images")
    # Đường dẫn đến thư mục output cho annotations mới
    output_dir = Path("data/processed/pdestre/head_annotations")
    output_dir.mkdir(parents=True, exist_ok=True)

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
        date_dir = ann_file.stem
        
        # Xử lý từng dòng trong file annotation
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {date_dir}", leave=False):
            # Tìm file ảnh head tương ứng
            person_id = int(row['id'])
            frame_num = int(row['frame'])
            
            # Tạo pattern tìm kiếm phù hợp với format thực tế
            pattern = f"{person_id}_{frame_num}_*_head.jpg"
            
            # Đường dẫn đến thư mục con chứa ảnh head cho ngày/video này
            current_head_dir = head_img_dir / date_dir
            
            # Tìm kiếm file ảnh bên trong thư mục con
            found = list(current_head_dir.glob(pattern))
            if found:
                # found[0] là đối tượng Path của file ảnh (full path)
                image_full_path = found[0]
                
                # Tính toán đường dẫn tương đối từ head_img_dir
                relative_image_path = image_full_path.relative_to(head_img_dir)
                
                all_data.append({
                    # Lưu đường dẫn tương đối dưới dạng string
                    'imagejpg': str(relative_image_path).replace('\\', '/'),
                    'beard': int(row['beard']),
                    'glasses': int(row['glasses'])
                })
                matched_annotations += 1

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
    create_head_annotations()
