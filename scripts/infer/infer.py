
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Import PyTorch trước tiên
import torch
import torch.nn as nn
from torchvision import transforms

# Sau đó mới import numpy và các thư viện khác
import matplotlib.pyplot as plt
from PIL import Image
import os

# Các import khác
from model import FeatureClassifier
from pathlib import Path

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
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    model.eval()
    return model

def preprocess_image(image_path, transform):
    """Tiền xử lý ảnh"""
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)
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

def visualize_prediction(image_path, predictions):
    """Hiển thị ảnh và kết quả dự đoán"""
    img = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(10, 5))
    
    # Hiển thị ảnh
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Input Image')
    
    # Hiển thị predictions
    plt.subplot(1, 2, 2)
    prediction_text = ""
    prediction_text += f"Gender: {GENDER_NAMES[predictions['gender'][0]]} ({predictions['gender'][1]:.2f})\n"
    prediction_text += f"Age: {AGE_NAMES[predictions['age'][0]]} ({predictions['age'][1]:.2f})\n"
    prediction_text += f"Ethnicity: {ETHNICITY_NAMES[predictions['ethnicity'][0]]} ({predictions['ethnicity'][1]:.2f})\n"
    prediction_text += f"Beard: {BEARD_NAMES[predictions['beard'][0]]} ({predictions['beard'][1]:.2f})\n"
    prediction_text += f"Glasses: {GLASSES_NAMES[predictions['glasses'][0]]} ({predictions['glasses'][1]:.2f})\n"
    prediction_text += f"Accessories: {ACCESSORIES_NAMES[predictions['accessories'][0]]} ({predictions['accessories'][1]:.2f})"
    
    plt.text(0.1, 0.5, prediction_text, fontsize=12)
    plt.axis('off')
    plt.title('Predictions')
    
    plt.tight_layout()
    plt.show()

def test_on_directory(model, directory_path, transform, device):
    """Test model trên toàn bộ thư mục"""
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Directory {directory} does not exist!")
        return
    
    # Lấy tất cả ảnh trong thư mục
    image_files = list(directory.glob('*.png'))
    print(f"Found {len(image_files)} images in {directory}")
    
    # Test trên 5 ảnh đầu tiên
    for img_path in image_files[:]:
        print(f"\nProcessing {img_path.name}")
        try:
            # Tiền xử lý ảnh
            img_tensor = preprocess_image(img_path, transform)
            
            # Dự đoán
            predictions = predict(model, img_tensor, device)
            
            # Hiển thị kết quả
            visualize_prediction(img_path, predictions)
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

def main():
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
    
    # Load model
    model_path = 'checkpoints/best_weight_combined.pth'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
    
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Test trên một thư mục
    test_dir = 'data/processed/hiep_dataset/images/MVI_9466/bbox_image/person_213'  # Thay đổi đường dẫn này
    print(f"\nTesting on directory: {test_dir}")
    test_on_directory(model, test_dir, transform, device)
    
    # # Test trên một ảnh cụ thể
    # test_image = 'outputs/images/im3.png'  # Thay đổi đường dẫn này
    # if os.path.exists(test_image):
    #     print(f"\nTesting on single image: {test_image}")
    #     img_tensor = preprocess_image(test_image, transform)
    #     predictions = predict(model, img_tensor, device)
    #     visualize_prediction(test_image, predictions)
    # else:
    #     print(f"Test image {test_image} not found!")

if __name__ == '__main__':
    main() 