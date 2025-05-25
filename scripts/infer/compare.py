import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
import torch
from torch.utils.data import DataLoader
from model import FeatureClassifier
import os
from pathlib import Path
from torchvision import transforms
from dataset_local import PdestreFeatureDataset
from train import collate_skip_none
from torch.utils.data import ConcatDataset
from tqdm import tqdm

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

# Danh sách features và class names tương ứng
FEATURES = ['gender', 'age', 'ethnicity', 'beard', 'glasses', 'accessories']
CLASS_NAMES = [GENDER_NAMES, AGE_NAMES, ETHNICITY_NAMES, BEARD_NAMES, GLASSES_NAMES, ACCESSORIES_NAMES]

def load_model(weight_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load model và weights"""
    try:
        model = FeatureClassifier()
        checkpoint = torch.load(weight_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Lỗi khi load model: {str(e)}")
        return None

def evaluate_model(model, test_loader, dataset_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Đánh giá model trên test dataset"""
    all_preds = [[] for _ in range(len(FEATURES))]
    all_labels = [[] for _ in range(len(FEATURES))]
    inference_times = []
    
    model.eval()
    
    # Thêm tqdm cho test_loader
    progress_bar = tqdm(test_loader, desc=f'Đánh giá trên {dataset_name}', 
                       total=len(test_loader), ncols=100)
    
    with torch.no_grad():
        for batch in progress_bar:
            if batch is None:
                continue
                
            images, features = batch
            images = images.to(device)
            
            # Đo thời gian inference
            start_time = time.time()
            outputs = model(images)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Chuyển đổi outputs thành predicted classes
            for i, feature_name in enumerate(FEATURES):
                # Lấy predictions cho feature hiện tại
                preds = outputs[feature_name]
                _, preds = torch.max(preds, 1)
                
                # Lấy ground truth cho feature hiện tại
                labels = features[:, i]  # Lấy cột tương ứng với feature hiện tại
                
                # Thêm vào list
                all_preds[i].extend(preds.cpu().numpy())
                all_labels[i].extend(labels.cpu().numpy())
            
            # Cập nhật mô tả của progress bar
            progress_bar.set_postfix({
                'batch_size': images.size(0),
                'avg_time': f"{sum(inference_times)/len(inference_times):.3f}s"
            })
    
    # Tính toán metrics cho từng feature
    results = []
    for i, feature in enumerate(FEATURES):
        accuracy = accuracy_score(all_labels[i], all_preds[i])
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels[i], all_preds[i], average='weighted')
        
        # Tạo confusion matrix
        cm = confusion_matrix(all_labels[i], all_preds[i])
        
        results.append({
            'feature': feature,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        })
    
    # Tính trung bình thời gian inference
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return {
        'dataset': dataset_name,
        'results': results,
        'avg_inference_time': avg_inference_time
    }

def plot_comparison(pdestre_results, custom_results):
    """Vẽ biểu đồ so sánh giữa hai models"""
    comparison_data = []
    
    for i, feature in enumerate(FEATURES):
        pdestre_metrics = pdestre_results['results'][i]
        custom_metrics = custom_results['results'][i]
        
        comparison_data.append({
            'feature': feature,
            'p-destre_accuracy': pdestre_metrics['accuracy'],
            'custom_accuracy': custom_metrics['accuracy'],
            'p-destre_f1': pdestre_metrics['f1'],
            'custom_f1': custom_metrics['f1']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(15, 10))
    
    # Accuracy comparison
    plt.subplot(2, 1, 1)
    x = np.arange(len(FEATURES))
    width = 0.35
    
    plt.bar(x - width/2, df['p-destre_accuracy'], width, label='P-DESTRE')
    plt.bar(x + width/2, df['custom_accuracy'], width, label='Custom Dataset')
    
    plt.xlabel('Features')
    plt.ylabel('Accuracy')
    plt.title('So sánh Accuracy giữa P-DESTRE và Custom Dataset')
    plt.xticks(x, FEATURES)
    plt.legend()
    
    # F1 comparison
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, df['p-destre_f1'], width, label='P-DESTRE')
    plt.bar(x + width/2, df['custom_f1'], width, label='Custom Dataset')
    
    plt.xlabel('Features')
    plt.ylabel('F1 Score')
    plt.title('So sánh F1 Score giữa P-DESTRE và Custom Dataset')
    plt.xticks(x, FEATURES)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()

def create_detailed_report(pdestre_results, custom_results):
    """Tạo báo cáo chi tiết về kết quả so sánh"""
    report = []
    
    for i, feature in enumerate(FEATURES):
        pdestre_metrics = pdestre_results['results'][i]
        custom_metrics = custom_results['results'][i]
        
        report.append({
            'Feature': feature,
            'P-DESTRE Accuracy': f"{pdestre_metrics['accuracy']:.4f}",
            'Custom Accuracy': f"{custom_metrics['accuracy']:.4f}",
            'P-DESTRE F1': f"{pdestre_metrics['f1']:.4f}",
            'Custom F1': f"{custom_metrics['f1']:.4f}",
            'P-DESTRE Precision': f"{pdestre_metrics['precision']:.4f}",
            'Custom Precision': f"{custom_metrics['precision']:.4f}",
            'P-DESTRE Recall': f"{pdestre_metrics['recall']:.4f}",
            'Custom Recall': f"{custom_metrics['recall']:.4f}"
        })
    
    # Thêm thông tin về thời gian inference
    report.append({
        'Feature': 'Inference Time',
        'P-DESTRE Accuracy': f"{pdestre_results['avg_inference_time']:.4f}s",
        'Custom Accuracy': f"{custom_results['avg_inference_time']:.4f}s",
        'P-DESTRE F1': '',
        'Custom F1': '',
        'P-DESTRE Precision': '',
        'Custom Precision': '',
        'P-DESTRE Recall': '',
        'Custom Recall': ''
    })
    
    # Tạo DataFrame và lưu thành CSV
    df = pd.DataFrame(report)
    df.to_csv('model_comparison_report.csv', index=False)
    print("\nBáo cáo chi tiết đã được lưu vào file 'model_comparison_report.csv'")

def main():
    # Đường dẫn đến các file weight
    pdestre_weight = "checkpoints/best_weight_6fts.pth"
    custom_weight = "checkpoints/best_weight_6fts.pth"
    
    # Đường dẫn đến dataset
    PDESTRE_JPG_DIR = Path("data/processed/images")
    PDESTRE_ANNOTATION_DIR = Path("data/processed/labels")  # Thư mục chứa các file .txt
    CUSTOM_JPG_DIR = Path("data/processed/images")
    CUSTOM_ANNOTATION_DIR = Path("data/processed/labels")  # Thư mục chứa các file .txt
    
    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_pdestre = load_model(pdestre_weight, device)
    model_custom = load_model(custom_weight, device)
    
    if model_pdestre is None or model_custom is None:
        print("Không thể load model. Vui lòng kiểm tra lại đường dẫn file weight.")
        return

    # Tạo transforms cho test
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Tạo dataset và dataloader cho P-DESTRE
    print("\n--- Loading P-DESTRE Test Dataset ---")
    # Tìm tất cả các file annotation trong thư mục
    pdestre_annotation_files = list(PDESTRE_ANNOTATION_DIR.glob('*.txt'))
    if not pdestre_annotation_files:
        print(f"Không tìm thấy file annotation nào trong {PDESTRE_ANNOTATION_DIR}")
        return
        
    print(f"Tìm thấy {len(pdestre_annotation_files)} file annotation")
    
    # Tạo dataset cho từng file annotation
    pdestre_datasets = []
    for ann_file in pdestre_annotation_files:
        try:
            dataset = PdestreFeatureDataset(
                jpg_dir=PDESTRE_JPG_DIR,
                annotation_file=ann_file,
                transform=test_transform
            )
            if len(dataset) > 0:
                pdestre_datasets.append(dataset)
                print(f"Đã load dataset từ file {ann_file.name} với {len(dataset)} mẫu")
        except Exception as e:
            print(f"Lỗi khi tạo dataset cho file {ann_file}: {e}")
            continue
    
    if not pdestre_datasets:
        print("Không thể tạo dataset P-DESTRE. Vui lòng kiểm tra lại dữ liệu.")
        return
        
    # Kết hợp các dataset
    pdestre_test_dataset = ConcatDataset(pdestre_datasets)
    
    pdestre_test_loader = DataLoader(
        pdestre_test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_skip_none
    )

    # Tạo dataset và dataloader cho Custom dataset
    print("\n--- Loading Custom Test Dataset ---")
    # Tìm tất cả các file annotation trong thư mục
    custom_annotation_files = list(CUSTOM_ANNOTATION_DIR.glob('*.txt'))
    if not custom_annotation_files:
        print(f"Không tìm thấy file annotation nào trong {CUSTOM_ANNOTATION_DIR}")
        return
        
    print(f"Tìm thấy {len(custom_annotation_files)} file annotation")
    
    # Tạo dataset cho từng file annotation
    custom_datasets = []
    for ann_file in custom_annotation_files:
        try:
            dataset = PdestreFeatureDataset(
                jpg_dir=CUSTOM_JPG_DIR,
                annotation_file=ann_file,
                transform=test_transform
            )
            if len(dataset) > 0:
                custom_datasets.append(dataset)
                print(f"Đã load dataset từ file {ann_file.name} với {len(dataset)} mẫu")
        except Exception as e:
            print(f"Lỗi khi tạo dataset cho file {ann_file}: {e}")
            continue
    
    if not custom_datasets:
        print("Không thể tạo dataset Custom. Vui lòng kiểm tra lại dữ liệu.")
        return
        
    # Kết hợp các dataset
    custom_test_dataset = ConcatDataset(custom_datasets)
    
    custom_test_loader = DataLoader(
        custom_test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_skip_none
    )
    
    print(f"\nP-DESTRE test dataset size: {len(pdestre_test_dataset)}")
    print(f"Custom test dataset size: {len(custom_test_dataset)}")
    
    # Đánh giá trên P-DESTRE test set
    print("\n--- Evaluating on P-DESTRE Test Set ---")
    pdestre_results = evaluate_model(model_pdestre, pdestre_test_loader, 'P-DESTRE', device)
    
    # Đánh giá trên custom test set
    print("\n--- Evaluating on Custom Test Set ---")
    custom_results = evaluate_model(model_custom, custom_test_loader, 'Custom Dataset', device)
    
    # So sánh kết quả
    plot_comparison(pdestre_results, custom_results)
    #plot_confusion_matrices(pdestre_results, custom_results)
    
    # Tạo báo cáo chi tiết
    create_detailed_report(pdestre_results, custom_results)

if __name__ == "__main__":
    main()