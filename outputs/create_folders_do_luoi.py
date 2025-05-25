import os

def create_sequential_mvi_folders(base_path, start_number, end_number):
    """
    Tạo một chuỗi các thư mục MVI_xxxx bên trong thư mục base_path.

    Args:
        base_path (str): Đường dẫn đến thư mục cha (ví dụ: "hiep_dataset").
        start_number (int): Số bắt đầu cho chuỗi thư mục (ví dụ: 9457).
        end_number (int): Số kết thúc cho chuỗi thư mục (ví dụ: 9467).
    """
    if not os.path.isdir(base_path):
        print(f"Lỗi: Thư mục '{base_path}' không tồn tại.")
        print("Vui lòng tạo thư mục này hoặc kiểm tra lại đường dẫn.")
        return

    print(f"Sẽ tạo các thư mục trong: '{os.path.abspath(base_path)}'")
    created_count = 0
    skipped_count = 0

    for i in range(start_number, end_number + 1): # +1 để bao gồm cả end_number
        folder_name = f"MVI_{i}"
        full_folder_path = os.path.join(base_path, folder_name)

        try:
            if not os.path.exists(full_folder_path):
                os.makedirs(full_folder_path) # Sử dụng makedirs để tạo nếu chưa có
                print(f"  Đã tạo: '{folder_name}'")
                created_count += 1
            else:
                print(f"  Bỏ qua (đã tồn tại): '{folder_name}'")
                skipped_count += 1
        except OSError as e:
            print(f"  Lỗi khi tạo thư mục '{folder_name}': {e}")

    print(f"\nHoàn thành. Đã tạo {created_count} thư mục mới, bỏ qua {skipped_count} thư mục đã tồn tại.")

if __name__ == "__main__":
    # --- Cấu hình ---

    path_to_hiep_dataset = "outputs/hiep_dataset"
    

    start_num = 9710
    end_num = 9719

    print(f"Bạn có chắc chắn muốn tạo các thư mục từ MVI_{start_num} đến MVI_{end_num}")
    print(f"bên trong thư mục '{os.path.abspath(path_to_hiep_dataset)}' không?")
    confirm = input("Nhập 'y' để tiếp tục: ")

    if confirm.lower() == 'y':
        create_sequential_mvi_folders(path_to_hiep_dataset, start_num, end_num)
    else:
        print("Thao tác đã bị hủy.")