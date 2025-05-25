import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

class HeadDetector:
    def __init__(self, model_path="yolov10n-face.pt"):
        """
        Khởi tạo HeadDetector với YOLOv10n face model
        """
        self.model = YOLO(model_path)

    def detect_head_from_person_crop(self, person_crop, conf_threshold=0.3):
        """
        Detect head từ ảnh đã được crop sẵn (person crop)
        Args:
            person_crop: Ảnh đã crop full body (numpy array)
            conf_threshold: Ngưỡng confidence cho face detection
        Returns:
            results: Dictionary chứa thông tin head detection
        """
        if person_crop is None or person_crop.size == 0:
            return {'success': False, 'error': 'Empty input crop'}

        # 1. Detect face trong crop
        face_boxes = self.detect_face_in_crop(person_crop, conf_threshold)

        if not face_boxes:
            return {'success': False, 'error': 'No face detected'}

        # 2. Lấy face có confidence cao nhất
        best_face = max(face_boxes, key=lambda x: x['confidence'])
        face_bbox = best_face['bbox']

        # 3. Mở rộng face thành head
        head_bbox_crop = self.expand_face_to_head(face_bbox, person_crop.shape)

        return {
            'success': True,
            'head_bbox': head_bbox_crop,
            'face_bbox': face_bbox,
            'confidence': best_face['confidence']
        }

    def detect_face_in_crop(self, cropped_image, conf_threshold=0.5):
        results = self.model(cropped_image, conf=conf_threshold, verbose=False)
        face_boxes = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confidences):
                if conf >= conf_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    face_boxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })
        return face_boxes

    def expand_face_to_head(self, face_bbox, crop_shape, expansion_factor=1.6):
        x1, y1, x2, y2 = face_bbox
        face_width = x2 - x1
        face_height = y2 - y1
        crop_h, crop_w = crop_shape[:2]
        width_expand = int(face_width * 0.3)
        head_x1 = max(0, x1 - width_expand)
        head_x2 = min(crop_w, x2 + width_expand)
        height_expand_top = int(face_height * 0.4)
        height_expand_bottom = int(face_height * 0.2)
        head_y1 = max(0, y1 - height_expand_top)
        head_y2 = min(crop_h, y2 + height_expand_bottom)
        return (head_x1, head_y1, head_x2, head_y2)

    def visualize_result(self, person_crop, result):
        img_vis = person_crop.copy()
        if result['success']:
            # Vẽ head bbox (màu đỏ)
            hx1, hy1, hx2, hy2 = result['head_bbox']
            cv2.rectangle(img_vis, (int(hx1), int(hy1)), (int(hx2), int(hy2)),
                         (0, 0, 255), 2)
            cv2.putText(img_vis, f'Head ({result["confidence"]:.2f})',
                       (int(hx1), int(hy1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Vẽ face bbox (màu vàng)
            fx1, fy1, fx2, fy2 = result['face_bbox']
            cv2.rectangle(img_vis, (int(fx1), int(fy1)), (int(fx2), int(fy2)),
                         (0, 255, 255), 1)
        return img_vis


def main():
    detector = HeadDetector("yolov10n-face.pt")

    # Load ảnh đã crop sẵn 
    crop_path = "data/processed/pdestre/images/13-11-2019-1-1/105/105_610_13112019_36.jpg" 
    person_crop = cv2.imread(crop_path)

    if person_crop is None:
        print("Không thể load ảnh crop!")
        return

    # Detect head
    result = detector.detect_head_from_person_crop(person_crop, conf_threshold=0.25)

    if result['success']:
        print(f"✅ Detected head successfully!")
        print(f"Head bbox (in crop): {result['head_bbox']}")
        print(f"Face confidence: {result['confidence']:.3f}")

        # Visualize
        img_result = detector.visualize_result(person_crop, result)

        # Hiển thị hoặc lưu kết quả
        cv2.imshow('Head Detection Result', img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Lưu kết quả
        cv2.imwrite('head_detection_result.jpg', img_result)

    else:
        print(f"❌ Failed to detect head: {result['error']}")

if __name__ == "__main__":
    main()