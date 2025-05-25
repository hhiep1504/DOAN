import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# Định nghĩa danh sách features (phải khớp với thứ tự trong tensor features)
FEATURES = ['gender', 'age', 'ethnicity', 'beard', 'glasses', 'accessories']

# Định nghĩa các class names (cần giữ nguyên như trong compare.py hoặc nơi khác)
GENDER_NAMES = {0: 'Nam', 1: 'Nữ', 2: 'Unknown'}
AGE_NAMES = {
    0: '0-17', 1: '18-55', 2: '55+', 3: 'Unknown'
}
ETHNICITY_NAMES = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Unknown'}
BEARD_NAMES = {0: 'Có râu', 1: 'Không râu', 2: 'Không rõ'}
GLASSES_NAMES = {0: 'Kính thường', 1: 'Kính râm', 2: 'Không kính', 3: 'Không rõ'}
ACCESSORIES_NAMES = {
    0: 'Túi xách', 1: 'Ba lô', 2: 'Túi kéo', 3: 'Ô/Dù', 
    4: 'Túi thể thao', 5: 'Túi đi chợ', 6: 'Không có', 7: 'Không rõ' # -1: 'Error' không có trong file annotation, nên bỏ qua
}

# Danh sách các dictionary class names tương ứng với FEATURES
CLASS_NAMES = [GENDER_NAMES, AGE_NAMES, ETHNICITY_NAMES, BEARD_NAMES, GLASSES_NAMES, ACCESSORIES_NAMES]

class CustomDataset(Dataset):
    def __init__(self, hiep_base_dir, annotation_dir, transform=None):
        """
        Dataset cho hiep_dataset với ảnh từ thư mục bbox_image.

        Args:
            hiep_base_dir (Path): Đường dẫn đến thư mục gốc của hiep_dataset thô (chứa các thư mục video).
            annotation_dir (Path): Đường dẫn đến thư mục chứa các file annotation .txt đã chuyển đổi.
            transform (callable, optional): Optional transform để áp dụng cho ảnh.
        """
        self.hiep_base_dir = hiep_base_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.samples = []

        # Các index của features trong dòng annotation .txt (0-based)
        # Dựa trên định dạng P-DESTRE đã phân tích:
        # frame, person_id, x, y, w, h, ?, ?, ?, ?, gender, age, ?, ethnicity, ?, ?, beard, ?, glasses, ?, ?, accessories, ?, ?
        # Index: 0, 1, 2, 3, 4, 5, ..., 10, 11, ..., 14, ..., 17, ..., 19, ..., 24, ...
        self.feature_indices = {
            'gender': 10,
            'age': 11,
            'ethnicity': 14,
            'beard': 17,
            'glasses': 19,
            'accessories': 24
        }

        self._load_annotations()

    def _load_annotations(self):
        """Đọc tất cả các file annotation .txt và lưu thông tin mẫu."""
        annotation_files = list(self.annotation_dir.glob('*.txt'))
        if not annotation_files:
            print(f"Không tìm thấy file annotation nào trong {self.annotation_dir}")
            return

        print(f"Tìm thấy {len(annotation_files)} file annotation trong {self.annotation_dir}")

        for ann_file in annotation_files:
            video_name = ann_file.stem # Tên video từ tên file annotation
            
            try:
                with open(ann_file, 'r') as f:
                    for line in f:
                        # Phân tích từng dòng
                        parts = line.strip().split(',')
                        if len(parts) != 26: # Kiểm tra số lượng cột
                            # print(f"Dòng không hợp lệ trong file {ann_file.name}: {line.strip()}") # Có thể bỏ comment để debug
                            continue
                            
                        try:
                            # Lấy thông tin cần thiết
                            frame_num = int(parts[0])
                            person_id = int(parts[1])
                            # Bbox [x, y, w, h] đã được làm tròn trong script chuyển đổi
                            bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                            
                            # Lấy các giá trị features
                            features_values = []
                            for i, feature_name in enumerate(FEATURES):
                                idx = self.feature_indices[feature_name]
                                
                                try:
                                    value = int(parts[idx])
                                    
                                    # Xử lý giá trị đặc biệt cho beard: ánh xạ 3 sang 2
                                    if feature_name == 'beard' and value == 3:
                                        value = 2 # Ánh xạ 3 (không hợp lệ) sang 2 (Không rõ)
                                        
                                    # Kiểm tra giá trị hợp lệ dựa trên CLASS_NAMES
                                    class_dict = CLASS_NAMES[i]
                                    if value not in class_dict and value != -1: # Cho phép giá trị -1 nếu có
                                         print(f"Cảnh báo: Giá trị thuộc tính không hợp lệ cho {feature_name}: {value} trong file {ann_file.name}, frame {frame_num}, person {person_id}. Dòng: {line.strip()}")
                                         # Ở đây ta vẫn lưu giá trị (sau khi xử lý trường hợp beard=3)
                                         # Việc xử lý giá trị không hợp lệ khác sẽ ở bước sau hoặc trong training
                                         
                                    features_values.append(value)
                                    
                                except (ValueError, IndexError) as e:
                                     print(f"Lỗi phân tích giá trị thuộc tính {feature_name} trong dòng: {line.strip()} - {e}")
                                     # Thêm giá trị mặc định nếu lỗi phân tích
                                     features_values.append(-1) # Sử dụng -1 để biểu thị lỗi/unknown
                                     # Không continue ở đây để cố gắng phân tích các features còn lại trong cùng dòng

                            # Kiểm tra xem có đủ 6 giá trị features không
                            if len(features_values) != len(FEATURES):
                                print(f"Lỗi: Số lượng features không khớp cho mẫu trong file {ann_file.name}, frame {frame_num}, person {person_id}. Dòng: {line.strip()}")
                                continue # Bỏ qua mẫu nếu thiếu features

                            # Lưu thông tin mẫu: (tên video, số frame, id người, bbox, giá trị features)
                            self.samples.append({
                                'video_name': video_name,
                                'frame_num': frame_num,
                                'person_id': person_id,
                                'bbox': bbox,
                                'features': features_values # List các giá trị features theo thứ tự trong FEATURES
                            })
                        except ValueError as e:
                            print(f"Lỗi phân tích dữ liệu trong dòng (frame, person_id, bbox): {line.strip()} - {e}")
                            continue
                        except Exception as e:
                             print(f"Lỗi không xác định khi xử lý dòng trong file {ann_file.name}: {e}")
                             continue

            except FileNotFoundError:
                print(f"Lỗi: File annotation không tồn tại tại {ann_file}")
                continue
            except Exception as e:
                print(f"Lỗi khi đọc hoặc xử lý file annotation {ann_file.name}: {e}")
                continue


        print(f"Đã load tổng cộng {len(self.samples)} mẫu từ các file annotation.")


    def __len__(self):
        """Trả về số lượng mẫu trong dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """Lấy một mẫu từ dataset tại index cho trước."""
        if idx >= len(self.samples):
            raise IndexError("Index out of bounds")

        sample_info = self.samples[idx]
        
        video_name = sample_info['video_name']
        frame_num = sample_info['frame_num']
        person_id = sample_info['person_id']
        # bbox = sample_info['bbox'] # Hiện tại không sử dụng bbox để cắt ảnh, đọc toàn bộ ảnh frame
        features_values = sample_info['features']

        # Tạo đường dẫn đến file ảnh trong thư mục bbox_image dựa trên cấu trúc mới
        # Cấu trúc: hiep_base_dir / video_name / "bbox_image" / "person_ID_NGƯỜI" / "person_ID_NGƯỜI_frame_SỐ_FRAME.png"
        person_folder_name = f"person_{person_id}"
        img_file_name = f"person_{person_id}_frame_{frame_num}.png" # Giả định file ảnh là PNG
        
        # Đường dẫn đầy đủ đến file ảnh
        img_path = self.hiep_base_dir / video_name / "bbox_image" / person_folder_name / img_file_name
        
        # Kiểm tra sự tồn tại của file ảnh
        if not img_path.exists():
             print(f"Lỗi: Không tìm thấy file ảnh cho mẫu {sample_info} tại đường dẫn {img_path}. Vui lòng kiểm tra lại cấu trúc thư mục và tên file ảnh trong thư mục bbox_image.")
             return None # Trả về None nếu không tìm thấy ảnh

        try:
            img = Image.open(img_path).convert('RGB')

            # Lưu trữ ảnh gốc trước khi apply transform nếu cần cho visualize
            original_img = img.copy() if self.transform else img

            if self.transform:
                img = self.transform(img)

            # Chuyển list features sang tensor
            features_tensor = torch.tensor(features_values, dtype=torch.long)

            # Trả về cả ảnh gốc (hoặc ảnh sau transform) và tensor features
            # Để visualize dễ dàng, ta trả về ảnh sau transform để phù hợp với đầu vào model
            return img, features_tensor, original_img, sample_info # Thêm original_img và sample_info để debug/visualize

        except Exception as e:
            print(f"Lỗi khi đọc hoặc xử lý ảnh {img_path}: {e}")
            return None

# Collate function để bỏ qua các mẫu None (khi getitem trả về None)
def collate_skip_none(batch):
    """
    Hàm collate để xử lý batch, loại bỏ các mẫu là None.
    Sử dụng khi dataset.__getitem__ có thể trả về None (ví dụ: do lỗi đọc file).
    """
    # Batch từ getitem bây giờ là list of (img, features_tensor, original_img, sample_info) hoặc None
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None # Trả về None nếu batch rỗng sau khi lọc

    # Tách các thành phần của batch
    imgs, features_tensors, original_imgs, sample_infos = zip(*batch)

    # Collate ảnh và features_tensor
    collated_imgs = torch.utils.data.dataloader.default_collate(imgs)
    collated_features = torch.utils.data.dataloader.default_collate(features_tensors)

    # Trả về batch đã collate, original_imgs và sample_infos (không collate)
    return collated_imgs, collated_features, list(original_imgs), list(sample_infos)

# Hàm visualize
def visualize_sample(dataset: CustomDataset, index: int = None):
    """
    Hiển thị ảnh và thuộc tính của một mẫu từ CustomDataset.

    Args:
        dataset (CustomDataset): Instance của CustomDataset.
        index (int, optional): Index của mẫu muốn visualize. Nếu None, chọn ngẫu nhiên.
    """
    if len(dataset) == 0:
        print("Dataset rỗng, không có mẫu nào để visualize.")
        return

    if index is None:
        index = random.randint(0, len(dataset) - 1)

    print(f"Visualizing sample at index: {index}")

    # Lấy mẫu từ dataset - getitem đã được sửa để trả về 4 giá trị
    sample = dataset[index]

    if sample is None:
        print(f"Không thể tải mẫu tại index {index}. Mẫu bị bỏ qua trong collate.")
        return

    img_tensor, features_tensor, original_img, sample_info = sample

    # Chuyển tensor ảnh về định dạng numpy để hiển thị (C, H, W) -> (H, W, C)
    # Nếu ảnh đã được normalize, việc hiển thị trực tiếp có thể không đúng màu
    # Để hiển thị đúng màu sắc ban đầu, sử dụng original_img (ảnh PIL)
    img_to_show = original_img

    plt.figure(figsize=(6, 6))
    plt.imshow(img_to_show)
    plt.title(f"Sample Index: {index}")
    plt.axis('off') # Tắt trục

    print("\n--- Features ---")
    feature_string = ""
    for i, feature_name in enumerate(FEATURES):
        label_value = features_tensor[i].item()
        class_dict = CLASS_NAMES[i]
        # Lấy tên lớp, sử dụng giá trị số nếu không tìm thấy trong dict
        label_name = class_dict.get(label_value, f"Unknown Value ({label_value})")
        feature_string += f"{feature_name.capitalize()}: {label_name}\n"
        print(f"{feature_name.capitalize()}: {label_name}")

    print("\n--- Sample Info ---")
    print(f"Video: {sample_info['video_name']}")
    print(f"Frame: {sample_info['frame_num']}")
    print(f"Person ID: {sample_info['person_id']}")
    print(f"Bbox (x, y, w, h): {sample_info['bbox']}")

    # Thêm thông tin features vào tiêu đề hoặc dưới ảnh nếu muốn
    # plt.xlabel(feature_string.strip()) # Có thể hiển thị features ở đây

    plt.show()

# Ví dụ cách sử dụng hàm visualize_sample (có thể thêm vào main hoặc chạy riêng)
if __name__ == "__main__":
    # Giả định các file annotation đã được tạo trong data/processed/hiep_dataset/annotations
    # và cấu trúc thư mục ảnh bbox_image đã tồn tại trong hiep_dataset gốc
    
    # Sửa đường dẫn HIEP_BASE_DIR để trỏ đến thư mục gốc của hiep_dataset gốc
    # Ví dụ: nếu hiep_dataset gốc nằm ở "hiep_dataset"
    HIEP_BASE_DIR = Path("data/processed/hiep_dataset/images") # <--- SỬA ĐƯỜNG DẪN NÀY
    
    # Đường dẫn đến thư mục chứa các file .txt đã chuyển đổi
    HIEP_ANNOTATION_DIR = Path("data/processed/hiep_dataset/annotations") 

    # Tạo transforms (có thể dùng transform đơn giản chỉ resize)
    test_transform_for_dataset = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Tạo dataset
    dataset_to_visualize = CustomDataset(
        hiep_base_dir=HIEP_BASE_DIR,
        annotation_dir=HIEP_ANNOTATION_DIR,
        transform=test_transform_for_dataset
    )

    # Visualize một vài mẫu
    print(f"Dataset size: {len(dataset_to_visualize)}")
    if len(dataset_to_visualize) > 0:
        # Visualize mẫu đầu tiên
        visualize_sample(dataset_to_visualize, index=0)

        # Visualize một mẫu ngẫu nhiên
        visualize_sample(dataset_to_visualize)

        # Visualize mẫu cuối cùng
        # visualize_sample(dataset_to_visualize, index=len(dataset_to_visualize) - 1)
    else:
        print("Không có mẫu nào trong dataset để visualize.")

    # Lưu ý: Nếu bạn chạy phần main này, nó chỉ hiển thị ảnh và in thông tin ra console.
    # Để tích hợp vào quy trình train/test, bạn sẽ tạo CustomDataset và DataLoader
    # trong script train/test của mình và sử dụng các dataloader đó.

