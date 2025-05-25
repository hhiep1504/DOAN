
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import time
from model import FeatureClassifier
from pathlib import Path
import argparse
import torch

# Định nghĩa các class names
GENDER_NAMES = {0: 'Nam', 1: 'Nữ', 2: 'Unknown'}
AGE_NAMES = {
    0: '0-11', 1: '12-17', 2: '18-24', 3: '25-34',
    4: '35-44', 5: '45-54', 6: '55-64', 7: '>65', 8: 'Unknown'
}
ETHNICITY_NAMES = {0: 'White', 1: 'Black', 2: 'Asian', 3: 'Indian', 4: 'Unknown'}
BEARD_NAMES = {0: 'Có râu', 1: 'Không râu', 2: 'Không rõ'}
GLASSES_NAMES = {0: 'Kính thường', 1: 'Kính râm', 2: 'Không kính', 3: 'Không rõ'}
ACCESSORIES_NAMES = {
    0: 'Túi xách', 1: 'Ba lô', 2: 'Túi kéo', 3: 'Ô/Dù', 
    4: 'Túi thể thao', 5: 'Túi đi chợ', 6: 'Không có', 7: 'Không rõ', -1: 'Error'
}

def load_model(model_path, device):
    """Load model từ checkpoint"""
    model = FeatureClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def preprocess_image(image, transform):
    """Tiền xử lý ảnh từ numpy array (BGR từ OpenCV)"""
    # Chuyển từ BGR sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Chuyển từ numpy array sang PIL Image
    pil_image = Image.fromarray(image_rgb)
    # Áp dụng transform
    img_tensor = transform(pil_image)
    return img_tensor.unsqueeze(0)  # Thêm batch dimension

def predict(model, image_tensor, device):
    """Dự đoán features từ ảnh"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        
        # Lấy class có xác suất cao nhất
        predictions = {}
        for key, output in outputs.items():
            probs = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            predictions[key] = (pred_class, confidence)
        
    return predictions

def draw_predictions(frame, predictions, x, y, w, h):
    """Vẽ dự đoán lên frame"""
    # Vẽ bounding box rõ hơn, với độ dày lớn hơn và màu nổi bật
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)  # Tăng độ dày đường viền
    
    # Thêm highlight cho bounding box để dễ nhìn hơn
    cv2.rectangle(frame, (x-2, y-2), (x + w+2, y + h+2), (255, 0, 0), 1)  # Thêm viền ngoài
    
    # Chuẩn bị text dự đoán
    texts = [
        f"Gender: {GENDER_NAMES[predictions['gender'][0]]} ({predictions['gender'][1]:.2f})",
        f"Age: {AGE_NAMES[predictions['age'][0]]} ({predictions['age'][1]:.2f})",
        f"Ethnicity: {ETHNICITY_NAMES[predictions['ethnicity'][0]]} ({predictions['ethnicity'][1]:.2f})",
        f"Beard: {BEARD_NAMES[predictions['beard'][0]]} ({predictions['beard'][1]:.2f})",
        f"Glasses: {GLASSES_NAMES[predictions['glasses'][0]]} ({predictions['glasses'][1]:.2f})",
        f"Accessories: {ACCESSORIES_NAMES[predictions['accessories'][0]]} ({predictions['accessories'][1]:.2f})"
    ]
    
    # Vẽ background cho text
    text_y = y - 10
    for text in texts:
        text_y += 20
        if text_y < y:  # Đảm bảo text không vẽ bên ngoài frame
            continue
        
        # Lấy kích thước text
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Vẽ background
        cv2.rectangle(frame, (x, text_y - text_height - 5), (x + text_width, text_y + 5), (0, 0, 0), -1)
        
        # Vẽ text
        cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def process_video(model, video_path, transform, device, output_path=None, detect_people=True, 
                  skip_frames=5, show_video=True, save_video=True):
    """Xử lý video và dự đoán người"""
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Không thể mở video {video_path}")
        return

    # Lấy thông số video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Tạo video writer nếu cần lưu
    out = None
    if save_video:
        if output_path is None:
            video_name = os.path.basename(video_path)
            base_name, ext = os.path.splitext(video_name)
            output_path = f"output_{base_name}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Load HOG detector nếu cần phát hiện người
    hog = None
    if detect_people:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("Loaded HOG detector for person detection")
    else:
        print("Person detection disabled, analyzing full frames")
    
    # Bắt đầu xử lý
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Xử lý mỗi skip_frames frame để tăng hiệu suất
        if frame_count % skip_frames != 0:
            if show_video:
                cv2.imshow('Video', frame)
            if save_video and out is not None:
                out.write(frame)
            # Dừng khi bấm 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Thay đổi kích thước frame nếu quá lớn để tăng tốc xử lý
        display_frame = frame.copy()
        
        if detect_people:
            # Phát hiện người trong frame
            boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
            
            # In thông tin về số người được phát hiện
            print(f"Frame {frame_count}: Detected {len(boxes)} people")
            total_detections += len(boxes)
            
            # Vẽ tất cả các bounding box trước khi phân tích
            for (x, y, w, h) in boxes:
                # Vẽ bounding box màu đỏ để kiểm tra
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Xử lý từng người được phát hiện
            for i, (x, y, w, h) in enumerate(boxes):
                # Lấy person crop
                person_crop = frame[y:y+h, x:x+w]
                if person_crop.size == 0:
                    print(f"Warning: Empty crop at ({x},{y},{w},{h})")
                    continue
                
                # Xử lý và dự đoán
                try:
                    img_tensor = preprocess_image(person_crop, transform)
                    predictions = predict(model, img_tensor, device)
                    
                    # Vẽ kết quả lên frame
                    display_frame = draw_predictions(display_frame, predictions, x, y, w, h)
                except Exception as e:
                    print(f"Error processing person {i} at ({x},{y},{w},{h}): {str(e)}")
        else:
            # Sử dụng toàn bộ frame nếu không phát hiện người
            try:
                img_tensor = preprocess_image(frame, transform)
                predictions = predict(model, img_tensor, device)
                
                # Vẽ kết quả lên toàn bộ frame
                x, y, w, h = 10, 10, width-20, height-20  # Để có thể nhìn thấy khung
                # Vẽ khung quanh cả frame
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                # Vẽ các dự đoán
                display_frame = draw_predictions(display_frame, predictions, x, y, w, h)
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
        
        # Hiển thị FPS
        fps_text = f"FPS: {(frame_count/(time.time() - start_time)):.2f}"
        cv2.putText(display_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Hiển thị frame đã xử lý
        if show_video:
            cv2.imshow('Video', display_frame)
        
        # Lưu frame đã xử lý
        if save_video and out is not None:
            out.write(display_frame)
        
        # Dừng khi bấm 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Giải phóng tài nguyên
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    # In thống kê
    total_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
    print(f"Average FPS: {frame_count/total_time:.2f}")
    if detect_people:
        print(f"Total people detected: {total_detections}")
    if save_video:
        print(f"Output saved to: {output_path}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Video feature inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="checkpoints/best_weight_6fts.pth", help="Path to model weights")
    parser.add_argument("--output", type=str, default=None, help="Path to output video")
    parser.add_argument("--detect", action="store_true", help="Detect people in video")
    parser.add_argument("--skip", type=int, default=5, help="Process every N frames")
    parser.add_argument("--no-show", action="store_true", help="Don't show video")
    parser.add_argument("--no-save", action="store_true", help="Don't save video")
    parser.add_argument("--detection-scale", type=float, default=1.05, help="HOG detection scale (smaller values detect more people but slower)")
    args = parser.parse_args()
    
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tạo transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Kiểm tra đường dẫn model
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found!")
        return
    
    # Load model
    print("Loading model...")
    model = load_model(args.model, device)
    
    # Kiểm tra đường dẫn video
    if not os.path.exists(args.video):
        print(f"Video file {args.video} not found!")
        return
    
    # Xử lý video
    print(f"Processing video: {args.video}")
    
    # Thiết lập thêm parameter nếu phát hiện người để điều chỉnh HOG detector
    if args.detect:
        # Tinh chỉnh HOG detector cho video
        process_video(
            model=model,
            video_path=args.video,
            transform=transform,
            device=device,
            output_path=args.output,
            detect_people=args.detect,
            skip_frames=args.skip,
            show_video=not args.no_show,
            save_video=not args.no_save,
            detection_scale=args.detection_scale
        )
    else:
        # Phân tích toàn bộ frame
        process_video(
            model=model,
            video_path=args.video,
            transform=transform,
            device=device,
            output_path=args.output,
            detect_people=args.detect,
            skip_frames=args.skip,
            show_video=not args.no_show,
            save_video=not args.no_save
        )

if __name__ == '__main__':
    main()