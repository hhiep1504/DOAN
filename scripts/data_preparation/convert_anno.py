import os
import re
import pathlib
import shutil # Thêm thư viện shutil để sao chép file hiệu quả

# --- Cấu hình ---
sampled_list_relative_path = "annotations/sampled_list.txt"
raw_annotation_dir = "data/raw/P-DESTRE/annotation"
output_labels_base_dir = "data/processed/labels" # Thư mục gốc lưu kết quả
path_prefix_in_list = "" # Để trống nếu đường dẫn trong list bắt đầu bằng ngày tháng

# --- Cache và Hàm lấy Annotation ---
# Cache để lưu trữ nội dung các tệp annotation đã đọc
# Key: tên tệp annotation (ví dụ: '08-11-2019-1-1.txt')
# Value: dictionary lồng nhau {frame_id: {person_id: annotation_line}}
annotation_cache = {}

def get_annotation_line(annotation_filename, target_frame_id, target_person_id):
    """
    Tìm dòng chú thích cụ thể trong tệp annotation gốc, sử dụng cache.
    """
    # Kiểm tra cache trước
    if annotation_filename not in annotation_cache:
        # Nếu chưa có trong cache, đọc tệp và xây dựng cache cho tệp đó
        annotation_cache[annotation_filename] = {}
        full_annotation_path = os.path.join(raw_annotation_dir, annotation_filename)
        try:
            with open(full_annotation_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    # Giả sử cột 0 là Frame ID, cột 1 là Person ID
                    if len(parts) >= 2:
                        try:
                            frame_id = int(parts[0])
                            person_id = int(parts[1])
                            # Lưu vào cache
                            if frame_id not in annotation_cache[annotation_filename]:
                                annotation_cache[annotation_filename][frame_id] = {}
                            # Ghi đè nếu có trùng lặp ID (hoặc xử lý khác nếu cần)
                            annotation_cache[annotation_filename][frame_id][person_id] = line
                        except ValueError:
                             pass # Bỏ qua dòng lỗi ID
        except FileNotFoundError:
            annotation_cache[annotation_filename] = None # Đánh dấu không tìm thấy tệp
            return None
        except Exception as e:
             print(f"Lỗi khi đọc tệp {full_annotation_path}: {e}")
             annotation_cache[annotation_filename] = None # Đánh dấu lỗi
             return None

    # Kiểm tra lại cache
    if annotation_cache[annotation_filename] is None:
         return None # Tệp không tồn tại hoặc lỗi đọc

    # Tìm trong cache
    if target_frame_id in annotation_cache[annotation_filename] and \
       target_person_id in annotation_cache[annotation_filename][target_frame_id]:
        return annotation_cache[annotation_filename][target_frame_id][target_person_id]

    return None # Không tìm thấy

# --- Xử lý chính ---
processed_count = 0
created_count = 0
not_found_count = 0
error_format_count = 0
error_id_count = 0
error_file_count = 0
error_other_count = 0

try:
    with open(sampled_list_relative_path, 'r') as infile:
        for line in infile:
            original_image_path_in_list = line.strip()
            if not original_image_path_in_list:
                continue

            processed_count += 1

            relative_image_path = original_image_path_in_list
            if path_prefix_in_list and relative_image_path.startswith(path_prefix_in_list):
                relative_image_path = relative_image_path[len(path_prefix_in_list):].lstrip('/')

            # Regex để trích xuất: date_folder, person_id_folder, person_id_file, frame_id, rest_of_filename.jpg
            match = re.match(r"([^/]+)/(\d+)/(\d+)_(\d+)_(.*\.jpg)", relative_image_path, re.IGNORECASE)

            if match:
                date_folder = match.group(1)
                person_id_folder = match.group(2)
                person_id_file = match.group(3)
                frame_id_str = match.group(4)
                image_filename_base = match.group(3) + "_" + match.group(4) + "_" + match.group(5) # Tên file ảnh gốc

                # Có thể thêm kiểm tra person_id_folder == person_id_file nếu cần

                person_id_str = person_id_file # Dùng ID người từ tên file
                annotation_filename_root = date_folder
                annotation_filename = f"{annotation_filename_root}.txt"

                try:
                    person_id = int(person_id_str)
                    frame_id = int(frame_id_str)

                    # Lấy dòng annotation cụ thể
                    annotation_line = get_annotation_line(annotation_filename, frame_id, person_id)

                    if annotation_line is not None:
                        # Tạo đường dẫn file label output
                        label_filename = os.path.splitext(image_filename_base)[0] + ".txt"
                        output_label_path = pathlib.Path(output_labels_base_dir) / date_folder / person_id_folder / label_filename

                        # Tạo thư mục cha nếu chưa có
                        output_label_path.parent.mkdir(parents=True, exist_ok=True)

                        # Ghi dòng annotation tìm được vào file label
                        with open(output_label_path, 'w') as outfile:
                            outfile.write(annotation_line + '\n')
                        created_count += 1
                    else:
                        # Xử lý không tìm thấy dòng annotation
                        raw_file_path = os.path.join(raw_annotation_dir, annotation_filename)
                        if not os.path.exists(raw_file_path):
                             error_file_count +=1
                             # print(f"Lỗi: Không tìm thấy tệp chú thích gốc: {raw_file_path} cho ảnh {original_image_path_in_list}")
                        else:
                             not_found_count += 1
                             # print(f"Không tìm thấy dòng chú thích cho: {original_image_path_in_list} (File: {annotation_filename}, Frame: {frame_id}, Person: {person_id})")

                except ValueError:
                     error_id_count += 1
                     # print(f"Lỗi: Không thể chuyển đổi ID thành số nguyên từ đường dẫn: {original_image_path_in_list}")
                except Exception as e:
                     error_other_count += 1
                     print(f"Lỗi không xác định khi xử lý dòng {original_image_path_in_list}: {e}")

            else:
                error_format_count += 1
                # print(f"Lỗi: Định dạng đường dẫn không khớp trong sampled_list.txt: {original_image_path_in_list}")

    print("\n--- Hoàn tất ---")
    print(f"Tổng số đường dẫn ảnh đã xử lý: {processed_count}")
    print(f"Số tệp label đã tạo: {created_count}")
    print(f"Số dòng không tìm thấy annotation tương ứng: {not_found_count}")
    print(f"Số lỗi không tìm thấy tệp annotation gốc: {error_file_count}")
    print(f"Số lỗi định dạng đường dẫn ảnh: {error_format_count}")
    print(f"Số lỗi chuyển đổi ID ảnh/người: {error_id_count}")
    print(f"Số lỗi khác: {error_other_count}")
    print(f"Các tệp label đã được tạo trong thư mục: '{output_labels_base_dir}'")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp danh sách mẫu tại '{sampled_list_relative_path}'")
except Exception as e:
    print(f"Đã xảy ra lỗi không mong muốn trong quá trình xử lý: {e}")

# --- Mã thay thế nếu không có shutil (đọc ghi thủ công) ---
# import os
# import re
# import pathlib

# # ... (Phần cấu hình và Bước 1 giữ nguyên) ...

# print("\nBước 2: Sao chép nội dung các tệp annotation gốc vào thư mục đích...")
# copied_files_count = 0
# error_reading_writing_count = 0
# skipped_missing_source_count = 0

# # Đảm bảo thư mục đích tồn tại
# pathlib.Path(output_labels_base_dir).mkdir(parents=True, exist_ok=True)

# for date_folder in required_dates:
#     source_annotation_filename = f"{date_folder}.txt"
#     source_annotation_path = os.path.join(raw_annotation_dir, source_annotation_filename)

#     destination_annotation_filename = f"{date_folder}.txt"
#     destination_annotation_path = os.path.join(output_labels_base_dir, destination_annotation_filename)

#     if os.path.exists(source_annotation_path):
#         try:
#             with open(source_annotation_path, 'r') as infile, open(destination_annotation_path, 'w') as outfile:
#                 for line in infile:
#                     outfile.write(line)
#             print(f"  Đã sao chép nội dung: {source_annotation_filename} -> {destination_annotation_path}")
#             copied_files_count += 1
#         except Exception as e:
#             print(f"  Lỗi khi đọc/ghi tệp {source_annotation_filename}: {e}")
#             error_reading_writing_count += 1
#     else:
#         print(f"  Bỏ qua: Không tìm thấy tệp nguồn {source_annotation_path}")
#         skipped_missing_source_count += 1

# # ... (Phần in kết quả tương tự) ...

# --- Xử lý ---
processed_dates_count = 0
consolidated_files_count = 0
errors_reading_files = 0
errors_writing_files = 0
errors_deleting_dirs = 0
dates_with_no_txt_files = 0

print(f"Bắt đầu gom các tệp annotation trong: {output_labels_base_dir}")

# Lấy danh sách các thư mục con trực tiếp (đây là các thư mục ngày)
try:
    date_folders = [d for d in os.listdir(output_labels_base_dir)
                    if os.path.isdir(os.path.join(output_labels_base_dir, d))]
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy thư mục '{output_labels_base_dir}'. Vui lòng kiểm tra lại đường dẫn.")
    exit()
except Exception as e:
    print(f"Lỗi khi truy cập thư mục '{output_labels_base_dir}': {e}")
    exit()

if not date_folders:
    print("Không tìm thấy thư mục ngày nào trong thư mục đích.")
    exit()

print(f"Tìm thấy {len(date_folders)} thư mục ngày để xử lý...")

for date_folder_name in date_folders:
    processed_dates_count += 1
    date_folder_path = pathlib.Path(output_labels_base_dir) / date_folder_name
    output_consolidated_file_path = pathlib.Path(output_labels_base_dir) / f"{date_folder_name}.txt"
    print(f"\n  Đang xử lý ngày: {date_folder_name}")

    all_annotation_lines = []
    found_txt_files = False

    # Tìm tất cả các file .txt một cách đệ quy trong thư mục ngày
    # Sử dụng rglob để tìm kiếm trong các thư mục con (person_id)
    for txt_file_path in date_folder_path.rglob('*.txt'):
        found_txt_files = True
        try:
            # Đọc nội dung (giả sử mỗi file chỉ có 1 dòng annotation quan trọng)
            with open(txt_file_path, 'r') as f:
                line = f.readline().strip() # Đọc dòng đầu tiên và loại bỏ khoảng trắng
                if line: # Chỉ thêm nếu dòng không rỗng
                    all_annotation_lines.append(line)
        except Exception as e:
            print(f"    Lỗi khi đọc tệp: {txt_file_path} - {e}")
            errors_reading_files += 1

    if not found_txt_files:
        print(f"    Cảnh báo: Không tìm thấy tệp .txt nào trong {date_folder_path}")
        dates_with_no_txt_files += 1
        continue # Chuyển sang thư mục ngày tiếp theo

    if not all_annotation_lines:
        print(f"    Cảnh báo: Không thu thập được dòng annotation hợp lệ nào từ {date_folder_path}")
        continue # Chuyển sang thư mục ngày tiếp theo nếu không có dòng nào


    # Ghi các dòng đã thu thập vào file tổng hợp mới
    try:
        with open(output_consolidated_file_path, 'w') as outfile:
            outfile.write("\n".join(all_annotation_lines) + "\n")
        print(f"    -> Đã tạo tệp tổng hợp: {output_consolidated_file_path.name}")
        consolidated_files_count += 1

        # (Tùy chọn) Xóa thư mục ngày cũ sau khi tạo file tổng hợp thành công
        try:
            shutil.rmtree(date_folder_path)
            print(f"    -> Đã xóa thư mục cũ: {date_folder_path}")
        except Exception as e:
            print(f"    Lỗi khi xóa thư mục {date_folder_path}: {e}")
            errors_deleting_dirs += 1

    except Exception as e:
        print(f"    Lỗi khi ghi tệp tổng hợp {output_consolidated_file_path.name}: {e}")
        errors_writing_files += 1


print("\n--- Hoàn tất ---")
print(f"Tổng số thư mục ngày đã xử lý: {processed_dates_count}")
print(f"Số tệp annotation tổng hợp đã tạo: {consolidated_files_count}")
print(f"Số thư mục ngày không chứa tệp .txt nào: {dates_with_no_txt_files}")
print(f"Số lỗi gặp phải khi đọc tệp con: {errors_reading_files}")
print(f"Số lỗi gặp phải khi ghi tệp tổng hợp: {errors_writing_files}")
print(f"Số lỗi gặp phải khi xóa thư mục cũ: {errors_deleting_dirs}")
print(f"Kết quả được lưu tại: '{output_labels_base_dir}'")