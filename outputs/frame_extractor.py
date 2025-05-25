import os
import cv2
import argparse
from tqdm import tqdm

def extract_frames_by_interval(video_path, output_dir, frame_interval):
    """
    Trích xuất các frame từ video theo một khoảng cách nhất định và lưu chúng vào thư mục đầu ra.

    Args:
        video_path (str): Đường dẫn đến file video đầu vào.
        output_dir (str): Thư mục để lưu các frame được trích xuất.
        frame_interval (int): Khoảng cách giữa các frame cần trích xuất 
                              (ví dụ: 5 nghĩa là lấy 1 frame sau mỗi 5 frame của video gốc).
    """
    if not os.path.exists(video_path):
        print(f"Lỗi: Không tìm thấy file video tại {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {video_path}")
        return

    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Lấy tên video để đặt tên file cho frame (không bao gồm phần mở rộng)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Đang xử lý video: {video_path}")
    print(f"Tổng số frame: {total_frames}, FPS: {fps}")
    print(f"Trích xuất 1 frame sau mỗi {frame_interval} frame.")
    print(f"Thư mục lưu frame: {output_dir}")

    count = 0
    saved_frame_count = 0
    with tqdm(total=total_frames, desc="Đang trích xuất frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Kết thúc video hoặc lỗi đọc frame

            if count % frame_interval == 0:
                # Lưu frame (OpenCV lưu ảnh ở định dạng BGR)
                frame_filename = os.path.join(output_dir, f"{video_name}_frame_{saved_frame_count:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
                saved_frame_count += 1
            
            count += 1
            pbar.update(1)

    cap.release()
    print(f"\nHoàn thành. Đã lưu {saved_frame_count} frame vào thư mục {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất frame từ video theo khoảng cách nhất định.")
    parser.add_argument("--video_path", type=str, default= 'custom_dataset/vid1.mp4', required=True, 
                        help="Đường dẫn đến file video đầu vào.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Thư mục để lưu các frame được trích xuất.")
    parser.add_argument("--frame_interval", type=int, default=5, 
                        help="Khoảng cách frame để trích xuất (mặc định: 5). Ví dụ: 5 nghĩa là lấy 1 frame sau mỗi 5 frame.")
    
    args = parser.parse_args()
    
    extract_frames_by_interval(args.video_path, args.output_dir, args.frame_interval)
