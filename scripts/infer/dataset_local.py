import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision  # Cần cho việc hiển thị ảnh grid
import math
import warnings

class PdestreFeatureDataset(Dataset):
    def __init__(self, jpg_dir, annotation_file, transform=None):
        """
        Args:
            jpg_dir: Đường dẫn đến thư mục jpg_Extracted_PIDS
            annotation_file: File annotation (ví dụ: 13-11-2019-1-1.txt)
            transform: Các biến đổi ảnh
        """
        self.jpg_dir = Path(jpg_dir)
        self.annotation_file = Path(annotation_file)
        # print(f"Loading annotations from {annotation_file}") # Bớt in ra khi inspect
        try:
            # Cẩn thận: Nếu file trống hoặc lỗi, pandas có thể báo lỗi
            self.annotations = pd.read_csv(self.annotation_file, header=None)
            # print(f"Total annotations before filtering: {len(self.annotations)}")
        except pd.errors.EmptyDataError:
            print(f"Warning: Annotation file is empty: {self.annotation_file}")
            self.annotations = pd.DataFrame() # Tạo DataFrame rỗng
        except Exception as e:
            print(f"Error reading annotation file {self.annotation_file}: {e}")
            self.annotations = pd.DataFrame()

        if not self.annotations.empty:
            # Lọc bỏ các ID = -1 (untracked persons)
            # Kiểm tra xem cột 1 có tồn tại không
            if 1 in self.annotations.columns:
                 self.annotations = self.annotations[self.annotations[1] != -1]
            else:
                print(f"Warning: Column 1 (person_id) not found in {self.annotation_file}. Skipping filtering.")
            # print(f"Total annotations after filtering: {len(self.annotations)}")

        # Lưu lại transform để dùng sau (nếu cần un-normalize)
        self.provided_transform = transform

        if transform is None:
            # Sử dụng transform mặc định nếu không được cung cấp
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            # Lưu lại mean/std để có thể un-normalize khi hiển thị
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]
        else:
            self.transform = transform
            # Cố gắng trích xuất mean/std từ transform được cung cấp
            # Giả định Normalize là bước cuối cùng hoặc có trong Compose
            self.norm_mean = [0.485, 0.456, 0.406] # Giá trị mặc định nếu không tìm thấy
            self.norm_std = [0.229, 0.224, 0.225] # Giá trị mặc định nếu không tìm thấy
            if hasattr(transform, 'transforms'): # Check if it's a Compose object
                for t in transform.transforms:
                    if isinstance(t, transforms.Normalize):
                        self.norm_mean = t.mean
                        self.norm_std = t.std
                        break
            elif isinstance(transform, transforms.Normalize): # Check if it's Normalize itself
                 self.norm_mean = transform.mean
                 self.norm_std = transform.std


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if idx >= len(self.annotations):
            raise IndexError("Index out of bounds")

        row = self.annotations.iloc[idx]
        
        try:
            # Lấy thông tin từ annotation
            frame = int(row[0])
            person_id = int(row[1])
            
            # Lấy các features
            gender = int(row[10])
            age = int(row[11])
            age_mapped = self.map_age_labels_8_to_4(age)
            ethnicity = int(row[14])
            beard = int(row[17]) if pd.notna(row[17]) else 2
            glasses = int(row[19]) if pd.notna(row[19]) else 3
            accessories = int(row[24]) if pd.notna(row[24]) else 7
            
            # Tạo tensor chứa tất cả features
            features = torch.tensor([
                gender, age_mapped, ethnicity, beard, glasses, accessories
            ], dtype=torch.long)
            
            # Lấy ảnh
            date_folder = self.annotation_file.stem
            if date_folder.startswith('._'):
                date_folder = date_folder[2:]
            elif date_folder.startswith('.'):
                date_folder = date_folder[1:]
            
            person_dir = self.jpg_dir / date_folder / str(person_id)
            date_str = self.get_date_str()
            
            if date_str == "UNKNOWN_DATE":
                return None
            
            pattern = f"{person_id}_{frame}_{date_str}_*.jpg"
            matching_images = sorted(list(person_dir.glob(pattern)))
            
            if not matching_images:
                return None
            
            img_path = matching_images[0]
            img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            return img, features
            
        except Exception as e:
            print(f"Lỗi khi xử lý sample {idx}: {e}")
            return None

    def get_date_str(self):
        filename = self.annotation_file.stem
        if filename.startswith('._'):
            filename = filename[2:]
        elif filename.startswith('.'):
             filename = filename[1:]
        parts = filename.split('-')
        if len(parts) >= 3:
             # Đảm bảo các phần là số trước khi join
             try:
                 day = int(parts[0])
                 month = int(parts[1])
                 year = int(parts[2])
                 # Format thành DDMMYYYY
                 return f"{day:02d}{month:02d}{year}"
             except ValueError:
                  warnings.warn(f"Could not parse date parts as integers from filename: {self.annotation_file.name}")
                  return "UNKNOWN_DATE"
        else:
             warnings.warn(f"Could not parse date from filename: {self.annotation_file.name}")
             return "UNKNOWN_DATE"

    def map_age_labels_8_to_4(self, age_label_8_classes):
        # Ánh xạ từ 8 lớp (0-7, 8: Unknown) sang 4 lớp (0, 1, 2, 3: Unknown)
        if age_label_8_classes in [0, 1]: # 0-11, 12-17 -> 0-17
            return 0
        elif age_label_8_classes in [2, 3, 4, 5]: # 18-24, 25-34, 35-44, 45-54 -> 18-55
            return 1
        elif age_label_8_classes in [6, 7]: # 55-64, >65 -> 55+
            return 2
        elif age_label_8_classes == 8: # Unknown
             return 3
        else:
            # Xử lý các giá trị không mong đợi, có thể trả về Unknown (3) hoặc -1
            return 3 # Mặc định về Unknown

# ======================================================================
# Helper function to un-normalize and display images
# ======================================================================
def imshow(inp, title=None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Hiển thị ảnh Tensor đã được normalize."""
    inp = inp.numpy().transpose((1, 2, 0)) # Chuyển từ [C, H, W] sang [H, W, C]
    mean = torch.tensor(mean).numpy()
    std = torch.tensor(std).numpy()
    inp = std * inp + mean # Un-normalize
    inp = torch.clip(torch.from_numpy(inp), 0, 1).numpy() # Đảm bảo giá trị pixel trong khoảng [0, 1]
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # Pause một chút để cập nhật plots

# ======================================================================
# Main execution block
# ======================================================================
if __name__ == "__main__":
    # --- Configuration ---
    # !!! THAY ĐỔI CÁC ĐƯỜNG DẪN NÀY CHO PHÙ HỢP VỚI MÁY CỦA BẠN !!!
    JPG_DIR = Path("/kaggle/input/pdestre-filtered/processed/processed/images")  # Đường dẫn đến thư mục chứa các thư mục con theo ngày
    ANNOTATION_DIR = Path("/kaggle/input/pdestre-filtered/processed/processed/labels") # Đường dẫn đến thư mục chứa các file .txt

    BATCH_SIZE = 8       # Số lượng ảnh hiển thị cùng lúc
    NUM_SAMPLES_TO_SHOW = 16 # Tổng số mẫu muốn xem
    NUM_WORKERS = 0       # Số worker cho DataLoader (0 thường dễ debug hơn)

    # --- Find annotation files ---
    all_annotation_files = sorted(list(ANNOTATION_DIR.glob('*.txt')))

    if not all_annotation_files:
        print(f"Error: No .txt files found in {ANNOTATION_DIR}")
        exit() # Thoát nếu không tìm thấy file nào

    print(f"Found {len(all_annotation_files)} annotation files in {ANNOTATION_DIR}")

    # --- Create list of Datasets ---
    list_of_datasets = []
    print("Attempting to load datasets...")
    # Sử dụng transform mặc định của class PdestreFeatureDataset
    # Nếu bạn có transform riêng, định nghĩa nó ở đây và truyền vào `transform=`
    dataset_transform = None

    for ann_file_path in tqdm(all_annotation_files, desc="Loading individual datasets"):
        try:
            # Tạo instance Dataset cho từng file annotation
            dataset_instance = PdestreFeatureDataset(
                jpg_dir=JPG_DIR,
                annotation_file=ann_file_path,
                transform=dataset_transform # Sử dụng transform đã định nghĩa (hoặc None)
            )
            # Chỉ thêm vào list nếu dataset không rỗng
            if len(dataset_instance) > 0:
                list_of_datasets.append(dataset_instance)
            else:
                 print(f"Info: Skipping empty dataset from {ann_file_path.name}")

        except Exception as e:
            # Ghi lại lỗi nhưng vẫn tiếp tục với các file khác
            print(f"\nError creating dataset for {ann_file_path.name}: {e}. Skipping this file.")

    # --- Combine Datasets ---
    if not list_of_datasets:
        print("\nError: No valid datasets could be loaded. Exiting.")
        exit()

    combined_dataset = ConcatDataset(list_of_datasets)
    print(f"\nSuccessfully loaded {len(list_of_datasets)} dataset(s).")
    print(f"Total combined dataset size: {len(combined_dataset)} samples.")

    # Lấy mean/std từ dataset đầu tiên để un-normalize (giả định chúng giống nhau)
    # Cần kiểm tra `list_of_datasets` không rỗng trước khi truy cập
    norm_mean = list_of_datasets[0].norm_mean
    norm_std = list_of_datasets[0].norm_std
    print(f"Using Mean: {norm_mean}, Std: {norm_std} for display.")


    # --- Create DataLoader ---
    # shuffle=False để xem các mẫu theo thứ tự, dễ kiểm tra hơn
    dataloader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    print(f"\nCreated DataLoader with batch size {BATCH_SIZE}.")

    # --- Iterate and Show Samples ---
    # print(f"Displaying first {NUM_SAMPLES_TO_SHOW} samples...")
    # samples_shown = 0
    # plt.figure(figsize=(15, int(15 * math.ceil(NUM_SAMPLES_TO_SHOW/BATCH_SIZE) / BATCH_SIZE ))) # Điều chỉnh kích thước figure

    # for i, batch in enumerate(dataloader):
    #     # DataLoader có thể trả về None nếu có lỗi và dùng collate_fn tùy chỉnh
    #     # Nhưng với collate mặc định, lỗi trong __getitem__ thường sẽ dừng chương trình
    #     # nếu không được bắt trong vòng lặp này
    #     try:
    #         images, features = batch # Giải nén batch

    #         # Tạo grid ảnh từ batch
    #         out = torchvision.utils.make_grid(images, nrow=int(math.sqrt(BATCH_SIZE))) # Hiển thị dạng lưới vuông

    #         # Tạo tiêu đề cho cả batch (ví dụ: gender của các ảnh)
    #         # Cẩn thận khi truy cập features, đảm bảo key tồn tại
    #         titles = []
    #         for idx in range(images.shape[0]):
    #              g = features['gender'][idx].item() if 'gender' in features else 'N/A'
    #              a = features['age'][idx].item() if 'age' in features else 'N/A'
    #              e = features['ethnicity'][idx].item() if 'ethnicity' in features else 'N/A'
    #              titles.append(f"G:{g} A:{a} E:{e}")

    #         batch_title = f"Batch {i+1}\n" + "\n".join([f"{k}: {v.tolist()}" for k,v in features.items()])

    #         print(f"\n--- Batch {i+1} ---")
    #         print(f"Images shape: {images.shape}")
    #         for k,v in features.items():
    #              print(f"Features '{k}' shape: {v.shape}, sample values: {v[:4].tolist()}...") # In vài giá trị đầu

    #         # Hiển thị grid ảnh và features
    #         imshow(out, title=batch_title, mean=norm_mean, std=norm_std)
    #         plt.show() # Hiển thị cửa sổ plot

    #         samples_shown += images.shape[0]
    #         if samples_shown >= NUM_SAMPLES_TO_SHOW:
    #             print(f"\nDisplayed {samples_shown} samples. Stopping.")
    #             break

    #     except Exception as e:
    #          print(f"\nError processing batch {i+1}: {e}")
    #          # Bạn có thể quyết định dừng lại hoặc tiếp tục với batch tiếp theo
    #          # break # Dừng lại nếu có lỗi
    #          continue # Bỏ qua batch lỗi và thử batch tiếp theo

    print("\nInspection finished.")


'''
Script sẽ in ra thông tin về số lượng file annotation tìm thấy.
Nó sẽ hiển thị tiến trình tải các dataset con.
Nó sẽ in ra tổng kích thước của dataset kết hợp.
Sau đó, nó sẽ mở một cửa sổ Matplotlib và hiển thị các batch ảnh (số lượng ảnh mỗi batch bằng BATCH_SIZE). Dưới mỗi batch ảnh, nó sẽ in ra thông tin về shape của tensor ảnh và các tensor features, cùng với một vài giá trị features đầu tiên của batch đó trong terminal.
Nó sẽ tiếp tục hiển thị các batch cho đến khi đủ NUM_SAMPLES_TO_SHOW ảnh được hiển thị hoặc hết dữ liệu.
'''