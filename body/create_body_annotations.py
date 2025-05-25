import pandas as pd
import os
from pathlib import Path
import glob

def create_pdestre_body_annotations_with_accessories():
    # Đường dẫn đến thư mục chứa các file fullbody annotations gốc P-DESTRE (.txt)
    fullbody_annotations_dir = 'data/processed/pdestre/annotations'

    # Đường dẫn đến file body annotations CSV P-DESTRE đã có (theo user cung cấp)
    existing_body_csv_path = 'data/body/pdestre/body_annotations/body_annotations.csv'

    # Đường dẫn lưu file body annotations CSV P-DESTRE cuối cùng
    # Lưu cùng thư mục với file gốc, có thể ghi đè nếu muốn
    output_body_csv_path = 'data/body/pdestre/body_annotations/body_annotations_with_accessories.csv'

    # Đảm bảo thư mục output tồn tại
    Path(output_body_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Đọc file body annotations CSV P-DESTRE đã có
    try:
        body_df = pd.read_csv(existing_body_csv_path)
        print(f"Đã đọc file body annotations P-DESTRE hiện có: {existing_body_csv_path} ({len(body_df)} dòng)")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file body annotations P-DESTRE tại {existing_body_csv_path}")
        print("Vui lòng đảm bảo file này đã được tạo ra trước đó.")
        return

    # Đọc tất cả các file fullbody annotation gốc P-DESTRE (.txt)
    all_fullbody_files = list(Path(fullbody_annotations_dir).glob('*.txt'))

    fullbody_list = []

    print(f"Tìm thấy {len(all_fullbody_files)} file annotation gốc P-DESTRE (.txt)")

    # Đọc và kết hợp dữ liệu từ các file fullbody annotation gốc
    for fullbody_file in all_fullbody_files:
        # Đối với P-DESTRE, tên file .txt thường là tên video hoặc ngày
        video_name = Path(fullbody_file).stem # Lấy tên file không có đuôi (ví dụ: 13-11-2019-1-1)

        df = pd.read_csv(fullbody_file, header=None)
        df.columns = [
            'frame', 'id', 'x', 'y', 'h', 'w', 'flag', 'yaw', 'pitch', 'roll',
            'gender', 'age', 'height', 'body_volume', 'ethnicity', 'hair_color',
            'hairstyle', 'beard', 'moustache', 'glasses', 'head_accessories',
            'upper_clothing', 'lower_clothing', 'feet', 'accessories', 'action'
        ]

        # Thêm cột tên video/ngày, id, frame để dùng cho merge
        df['video_name'] = video_name
        df[['id', 'frame']] = df[['id', 'frame']].astype(int) # Đảm bảo kiểu dữ liệu

        # Chỉ giữ các cột cần thiết và thêm vào list
        fullbody_list.append(df[['video_name', 'frame', 'id', 'accessories']])

    # Gộp tất cả fullbody annotations lại
    if not fullbody_list:
         print("Không có dữ liệu từ file annotation gốc P-DESTRE (.txt) để xử lý.")
         return

    fullbody_df = pd.concat(fullbody_list, ignore_index=True)
    print(f"Đã gộp {len(fullbody_df)} dòng từ các file annotation gốc P-DESTRE")

    # --- Debug Print ---
    print("\nSample data from fullbody_df (from original .txt files):")
    print(fullbody_df.head())
    # --- End Debug Print ---

    # --- Tạo DataFrame tóm tắt accessories cho mỗi cặp (video_name, id) ---
    # Lấy giá trị accessories đầu tiên hoặc phổ biến nhất cho mỗi (video_name, id)
    # Ở đây tôi sẽ lấy giá trị đầu tiên tìm thấy cho mỗi cặp (video_name, id)
    # Nếu accessories có thể thay đổi, bạn có thể cần logic phức tạp hơn (vd: mode)
    accessories_summary = fullbody_df.groupby(['video_name', 'id'])['accessories'].first().reset_index()

    # Mapping accessories về 4 nhóm (áp dụng cho accessories_summary)
    accessories_mapping = {
        0: 0,  # Bag
        1: 1,  # Backpack
        2: 2,  # Rolling Bag -> Other
        3: 2,  # Umbrella -> Other
        4: 2,  # Sport Bag -> Other
        5: 2,  # Market Bag -> Other
        6: 3,  # Nothing
        7: 2   # Unknown -> Other
    }
    accessories_summary['accessories_mapped'] = accessories_summary['accessories'].apply(lambda x: accessories_mapping.get(x, 2)).astype(int)
    accessories_summary.drop(columns=['accessories'], inplace=True) # Bỏ cột accessories gốc
    # --- End Tạo DataFrame tóm tắt ---


    # Trích xuất video_name/ngày, id, và frame từ image_path trong body_df hiện có
    def extract_info_from_pdestre_image_path(image_path):
        try:
            parts = Path(image_path).parts
            if len(parts) >= 3:
                video_name = parts[0]
                person_dir_name = parts[1]
                # filename_stem = Path(parts[-1]).stem # Không cần stem nữa

                try:
                    person_id = int(person_dir_name)
                except ValueError:
                    # print(f"Warning: Could not parse person_id from directory name: {person_dir_name} in {image_path}")
                    person_id = None

                if video_name is not None and person_id is not None:
                    return video_name, person_id
                else:
                    # print(f"Warning: Missing required info (video_name or person_id) from image path {image_path} after parsing.")
                    return None, None
            else:
                 # print(f"Warning: Unexpected image path format for parsing P-DESTRE: {image_path}")
                 return None, None
        except (AttributeError) as e:
            # print(f"Warning: Could not process image path {image_path} - Error: {e}")
            return None, None


    # Áp dụng hàm trích xuất thông tin và tạo cột tạm trong body_df
    print("\nĐang trích xuất thông tin video/id từ đường dẫn ảnh body P-DESTRE...")
    # Chỉ trích xuất video_name và id
    extracted_info = body_df['image_path'].apply(extract_info_from_pdestre_image_path)

    # Chuyển kết quả sang DataFrame và gán vào cột tạm
    extracted_info_df = pd.DataFrame(extracted_info.tolist(), columns=['temp_video_name', 'temp_id'])
    body_df = pd.concat([body_df, extracted_info_df], axis=1)

    # Loại bỏ các dòng không trích xuất được video_name hoặc id
    body_df.dropna(subset=['temp_video_name', 'temp_id'], inplace=True)
    body_df[['temp_id']] = body_df[['temp_id']].astype(int) # Đảm bảo kiểu int cho merge

    print(f"Đã trích xuất thông tin thành công cho {len(body_df)} dòng")

    # --- Debug Print ---
    print("\nSample data from body_df (after extracting info from image_path):")
    print(body_df[['image_path', 'temp_video_name', 'temp_id']].head())
    print("\nSample data from accessories_summary:")
    print(accessories_summary.head())
    # --- End Debug Print ---


    # Merge body_df với accessories_summary
    print("\nĐang merge dữ liệu body annotations P-DESTRE với thông tin accessories tóm tắt...")
    merged_df = pd.merge(
        body_df,
        accessories_summary,
        left_on=['temp_video_name', 'temp_id'],
        right_on=['video_name', 'id'],
        how='left'
    )

    # Bỏ các cột tạm và cột merge không cần thiết
    merged_df.drop(columns=['temp_video_name', 'temp_id', 'video_name', 'id'], inplace=True)
    merged_df.rename(columns={'accessories_mapped': 'accessories'}, inplace=True)

    # Xử lý các dòng không tìm thấy accessories sau merge
    merged_df['accessories'].fillna(2, inplace=True)
    merged_df['accessories'] = merged_df['accessories'].astype(int)

    # Kiểm tra và lọc các ảnh không tồn tại
    print("\nKiểm tra sự tồn tại của các file ảnh...")
    body_images_dir = Path('data/body/pdestre/body_images')
    valid_rows = []
    missing_images = []

    for idx, row in merged_df.iterrows():
        img_path = body_images_dir / row['image_path']
        if img_path.exists():
            valid_rows.append(row)
        else:
            missing_images.append(row['image_path'])

    if missing_images:
        print(f"\nTìm thấy {len(missing_images)} ảnh không tồn tại:")
        for img in missing_images[:5]:
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... và {len(missing_images) - 5} ảnh khác")

    # Cập nhật DataFrame với chỉ các dòng hợp lệ
    merged_df = pd.DataFrame(valid_rows)
    print(f"\nSố lượng mẫu sau khi lọc: {len(merged_df)}")

    # Lưu file CSV body annotation cuối cùng
    merged_df.to_csv(output_body_csv_path, index=False)

    print(f"\nĐã cập nhật file body annotations P-DESTRE với cột 'accessories' tại: {output_body_csv_path}")
    print(f"Tổng số dòng trong file cuối cùng: {len(merged_df)}")

    # In thống kê về số lượng mẫu cho mỗi lớp accessories
    accessories_counts = merged_df['accessories'].value_counts()
    print("\nSố lượng mẫu cho mỗi lớp accessories:")
    for label, count in accessories_counts.items():
        print(f"  Lớp {label}: {count} mẫu")

# ------------------------------------------------------------------------------
# Hàm xử lý Dataset Hiep
# ------------------------------------------------------------------------------

def create_hiep_body_annotations_with_accessories():
    # Đường dẫn đến thư mục chứa các file fullbody annotations gốc Hiep (.txt)
    fullbody_annotations_dir = 'data/processed/hiep_dataset/annotations' # Cần kiểm tra lại đường dẫn này

    # Đường dẫn đến file body annotations CSV Hiep đã có
    # Dựa vào log output trước: data/body/hiep_dataset/body_annotations/body_annotations.csv
    existing_body_csv_path = 'data/body/hiep_dataset/body_annotations/body_annotations.csv'

    # Đường dẫn lưu file body annotations CSV Hiep cuối cùng
    # Lưu cùng thư mục với file gốc, có thể ghi đè nếu muốn
    output_body_csv_path = 'data/body/hiep_dataset/body_annotations/body_annotations_with_accessories.csv'

    # Đảm bảo thư mục output tồn tại
    Path(output_body_csv_path).parent.mkdir(parents=True, exist_ok=True)

    # Đọc file body annotations CSV Hiep đã có
    try:
        body_df = pd.read_csv(existing_body_csv_path)
        print(f"Đã đọc file body annotations Hiep hiện có: {existing_body_csv_path} ({len(body_df)} dòng)")
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file body annotations Hiep tại {existing_body_csv_path}")
        print("Vui lòng đảm bảo file này đã được tạo ra trước đó.")
        return

    # Đọc tất cả các file fullbody annotation gốc Hiep (.txt)
    all_fullbody_files = list(Path(fullbody_annotations_dir).glob('*.txt'))

    fullbody_list = []

    print(f"Tìm thấy {len(all_fullbody_files)} file annotation gốc Hiep (.txt)")

    # Đọc và kết hợp dữ liệu từ các file fullbody annotation gốc
    for fullbody_file in all_fullbody_files:
        video_name = Path(fullbody_file).stem # Lấy tên file không có đuôi (ví dụ: MVI_9421)

        df = pd.read_csv(fullbody_file, header=None)
        df.columns = [
            'frame', 'id', 'x', 'y', 'h', 'w', 'flag', 'yaw', 'pitch', 'roll',
            'gender', 'age', 'height', 'body_volume', 'ethnicity', 'hair_color',
            'hairstyle', 'beard', 'moustache', 'glasses', 'head_accessories',
            'upper_clothing', 'lower_clothing', 'feet', 'accessories', 'action'
        ]

        # Thêm cột tên video, id, frame để dùng cho merge
        df['video_name'] = video_name
        df[['id', 'frame']] = df[['id', 'frame']].astype(int) # Đảm bảo kiểu dữ liệu

        # Chỉ giữ các cột cần thiết và thêm vào list
        fullbody_list.append(df[['video_name', 'frame', 'id', 'accessories']])

    if not fullbody_list:
         print("Không có dữ liệu từ file annotation gốc Hiep (.txt) để xử lý.")
         return

    fullbody_df = pd.concat(fullbody_list, ignore_index=True)
    print(f"Đã gộp {len(fullbody_df)} dòng từ các file annotation gốc Hiep")

    # --- Debug Print ---
    print("\nSample data from fullbody_df (from original .txt files - Hiep):")
    print(fullbody_df.head())
    # --- End Debug Print ---

    # Trích xuất video_name, id, và frame từ image_path trong body_df hiện có (cho Hiep)
    # image_path có dạng video_name/bbox_image/person_id/ten_file_ảnh.png
    # Ví dụ: MVI_9464/bbox_image/person_55/person_55_frame_37_body.png
    def extract_info_from_hiep_image_path(image_path):
        try:
            parts = Path(image_path).parts # Sử dụng Path.parts để xử lý trên các OS khác nhau
            if len(parts) >= 4:
                video_name = parts[0] # e.g., MVI_9464
                person_part = parts[2] # e.g., person_55
                filename_stem = Path(parts[3]).stem # e.g., person_55_frame_37_body

                try:
                    person_id = int(person_part.split('_')[1]) # e.g., 55 from person_55
                except (IndexError, ValueError):
                     print(f"Warning: Could not parse person_id from person part: {person_part} in {image_path}")
                     person_id = None

                # Tên file có dạng person_ID_frame_FRAME_body
                # Lấy phần tử thứ 2 từ cuối ('37' trong ['person', '55', 'frame', '37', 'body'])
                filename_parts = filename_stem.split('_')
                frame = None
                if len(filename_parts) >= 2:
                    try:
                        # Lấy phần tử cuối cùng là 'body', trước đó là frame number
                        # Hoặc dùng logic cũ lấy phần tử thứ 2 từ cuối (nếu tên file luôn có dạng person_ID_frame_FRAME)
                        # Logic cũ: frame_str = filename_parts[-2]; frame = int(frame_str)
                        # Logic mới dựa trên tên file person_ID_frame_FRAME_body: parts[-2] là FRAME
                        if len(filename_parts) >= 2: # Đảm bảo có ít nhất person_ID, frame_FRAME
                             frame = int(filename_parts[-2]) # Lấy frame number
                    except ValueError:
                          print(f"Warning: Could not parse frame from filename stem {filename_stem} in {image_path}")
                          frame = None


                if video_name is not None and person_id is not None and frame is not None:
                    return video_name, person_id, frame
                else:
                    print(f"Warning: Missing required info (video_name, person_id, or frame) from image path {image_path} after parsing.")
                    return None, None, None
            else:
                 print(f"Warning: Unexpected image path format for parsing Hiep: {image_path}")
                 return None, None, None
        except (AttributeError) as e:
            print(f"Warning: Could not process image path {image_path} - Error: {e}")
            return None, None, None


    # Áp dụng hàm trích xuất thông tin và tạo cột tạm trong body_df
    print("\nĐang trích xuất thông tin video/id/frame từ đường dẫn ảnh body Hiep...")
    extracted_info = body_df['image_path'].apply(extract_info_from_hiep_image_path)

    # Chuyển kết quả sang DataFrame và gán vào cột tạm
    extracted_info_df = pd.DataFrame(extracted_info.tolist(), columns=['temp_video_name', 'temp_id', 'temp_frame'])
    body_df = pd.concat([body_df, extracted_info_df], axis=1)

    # Loại bỏ các dòng không trích xuất được thông tin
    body_df.dropna(subset=['temp_video_name', 'temp_id', 'temp_frame'], inplace=True)
    body_df[['temp_id', 'temp_frame']] = body_df[['temp_id', 'temp_frame']].astype(int) # Đảm bảo kiểu int cho merge

    print(f"Đã trích xuất thông tin thành công cho {len(body_df)} dòng")

    # --- Debug Print ---
    print("\nSample data from body_df (after extracting info from image_path - Hiep):")
    print(body_df[['image_path', 'temp_video_name', 'temp_id', 'temp_frame']].head())
    # --- End Debug Print ---


    # Mapping accessories về 4 nhóm (áp dụng cho fullbody_df)
    accessories_mapping = {
        0: 0,  # Bag
        1: 1,  # Backpack
        2: 2,  # Rolling Bag -> Other
        3: 2,  # Umbrella -> Other
        4: 2,  # Sport Bag -> Other
        5: 2,  # Market Bag -> Other
        6: 3,  # Nothing
        7: 2   # Unknown -> Other
    }
    # Sử dụng .get để xử lý các giá trị accessories có thể không có trong mapping
    fullbody_df['accessories_mapped'] = fullbody_df['accessories'].apply(lambda x: accessories_mapping.get(x, 2)).astype(int)


    # Merge body_df với fullbody_df để thêm cột accessories_mapped
    print("\nĐang merge dữ liệu body annotations Hiep với thông tin accessories gốc...")
    merged_df = pd.merge(
        body_df,
        fullbody_df[['video_name', 'frame', 'id', 'accessories_mapped']],
        left_on=['temp_video_name', 'temp_frame', 'temp_id'],
        right_on=['video_name', 'frame', 'id'],
        how='left'
    )

    # Bỏ các cột tạm và cột merge không cần thiết
    merged_df.drop(columns=['temp_video_name', 'temp_id', 'temp_frame', 'video_name', 'frame', 'id'], inplace=True)
    merged_df.rename(columns={'accessories_mapped': 'accessories'}, inplace=True)

    # Xử lý các dòng không tìm thấy accessories sau merge
    merged_df['accessories'].fillna(2, inplace=True)
    merged_df['accessories'] = merged_df['accessories'].astype(int)

    # Kiểm tra và lọc các ảnh không tồn tại
    print("\nKiểm tra sự tồn tại của các file ảnh...")
    body_images_dir = Path('data/body/hiep_dataset/body_images')
    valid_rows = []
    missing_images = []

    for idx, row in merged_df.iterrows():
        img_path = body_images_dir / row['image_path']
        if img_path.exists():
            valid_rows.append(row)
        else:
            missing_images.append(row['image_path'])

    if missing_images:
        print(f"\nTìm thấy {len(missing_images)} ảnh không tồn tại:")
        for img in missing_images[:5]:
            print(f"  - {img}")
        if len(missing_images) > 5:
            print(f"  ... và {len(missing_images) - 5} ảnh khác")

    # Cập nhật DataFrame với chỉ các dòng hợp lệ
    merged_df = pd.DataFrame(valid_rows)
    print(f"\nSố lượng mẫu sau khi lọc: {len(merged_df)}")

    # Lưu file CSV body annotation cuối cùng
    merged_df.to_csv(output_body_csv_path, index=False)

    print(f"\nĐã cập nhật file body annotations Hiep với cột 'accessories' tại: {output_body_csv_path}")
    print(f"Tổng số dòng trong file cuối cùng: {len(merged_df)}")

    # In thống kê về số lượng mẫu cho mỗi lớp accessories
    accessories_counts = merged_df['accessories'].value_counts()
    print("\nSố lượng mẫu cho mỗi lớp accessories:")
    for label, count in accessories_counts.items():
        print(f"  Lớp {label}: {count} mẫu")


if __name__ == '__main__':
    # Chạy hàm tạo annotation cho P-DESTRE body
    create_pdestre_body_annotations_with_accessories()

    # Chạy hàm tạo annotation cho Hiep body
    #create_hiep_body_annotations_with_accessories() 