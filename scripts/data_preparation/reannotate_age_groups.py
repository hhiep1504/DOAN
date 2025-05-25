import os
import glob
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

"""
Script để annotation lại feature age trong các file annotation thành 3 nhóm mới:
- Nhóm 0: 0-17 (Trẻ em)
- Nhóm 1: 18-54 (Người trưởng thành)
- Nhóm 2: 55+ (Người cao tuổi)
- Nhóm 3: Không xác định (giữ nguyên)
"""

# Ánh xạ các nhóm tuổi cũ sang nhóm mới
AGE_GROUP_MAPPING = {
    0: 0,  # 0-11 -> 0-17
    1: 0,  # 12-17 -> 0-17
    2: 1,  # 18-24 -> 18-54
    3: 1,  # 25-34 -> 18-54
    4: 1,  # 35-44 -> 18-54
    5: 1,  # 45-54 -> 18-54
    6: 2,  # 55-64 -> 55+
    7: 2,  # >65 -> 55+
    8: 3,  # Unknown -> Unknown (để là 3 sau khi hoàn thành)
}

def reannotate_age_groups(annotation_dir, output_dir=None, backup=True):
    """
    Đọc tất cả các file annotation trong thư mục và thay đổi nhóm tuổi
    
    Args:
        annotation_dir: Thư mục chứa các file annotation
        output_dir: Thư mục đầu ra (nếu None, ghi đè lên file cũ)
        backup: Tạo bản sao lưu của các file gốc trước khi thay đổi
    """
    # Tạo thư mục đầu ra nếu cần
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = annotation_dir

    # Tạo thư mục backup nếu cần
    if backup:
        backup_dir = os.path.join(annotation_dir, 'backup_original')
        os.makedirs(backup_dir, exist_ok=True)
    
    # Tìm tất cả các file annotation (.txt)
    annotation_files = glob.glob(os.path.join(annotation_dir, '*.txt'))
    
    if not annotation_files:
        print(f"Không tìm thấy file annotation nào trong {annotation_dir}")
        return
    
    print(f"Tìm thấy {len(annotation_files)} file annotation")
    
    # Đọc và cập nhật từng file
    total_records = 0
    total_changed = 0
    
    for file_path in tqdm(annotation_files, desc="Xử lý file annotation"):
        file_name = os.path.basename(file_path)
        
        # Backup file gốc nếu cần
        if backup:
            backup_path = os.path.join(backup_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
        
        # Đọc file annotation
        try:
            lines = []
            changed_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Format của file annotation là csv
                    values = line.strip().split(',')
                    if len(values) >= 12:  # Cần ít nhất 12 cột để chứa feature age (index 11)
                        try:
                            # Lấy giá trị age hiện tại
                            current_age_group = int(float(values[11]))
                            
                            # Map sang nhóm mới
                            if current_age_group in AGE_GROUP_MAPPING:
                                new_age_group = AGE_GROUP_MAPPING[current_age_group]
                                if new_age_group != current_age_group:
                                    values[11] = str(new_age_group)
                                    changed_count += 1
                        except (ValueError, IndexError) as e:
                            print(f"Lỗi khi xử lý dòng trong file {file_name}: {e}")
                    
                    # Thêm dòng đã được xử lý (đã thay đổi hoặc giữ nguyên)
                    lines.append(','.join(values))
            
            # Ghi lại file
            output_path = os.path.join(output_dir, file_name)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            total_records += len(lines)
            total_changed += changed_count
            
        except Exception as e:
            print(f"Lỗi khi xử lý file {file_name}: {e}")
    
    print(f"Hoàn thành: Đã xử lý {total_records} bản ghi, thay đổi {total_changed} nhóm tuổi")
    
    

if __name__ == "__main__":
    # Đường dẫn đến thư mục chứa các file annotation
    ANNOTATION_DIR = "data/processed/labels"
    
    # Nếu muốn lưu vào thư mục khác, uncomment dòng dưới
    # OUTPUT_DIR = "data/raw/P-DESTRE/annotation_new_age"
    
    # Chạy hàm chính
    reannotate_age_groups(ANNOTATION_DIR, output_dir=None, backup=True) 