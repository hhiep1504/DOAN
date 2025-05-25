import os
import json
import numpy as np
from pathlib import Path
from train import Trainer

# Cấu hình training
config = {
    # Data
    'image_dir': 'data/head/hiep_dataset/head_images',
    'train_annotation': 'data/head/hiep_dataset/head_annotations/head_train.csv',
    'val_annotation': 'data/head/hiep_dataset/head_annotations/head_val.csv',
    'test_annotation': 'data/head/hiep_dataset/head_annotations/head_test.csv',
    
    # Model
    'backbone': 'resnet50',  # ['resnet50', 'efficientnet_b0', 'mobilenet_v3']
    'pretrained': True,
    'freeze_backbone': False,
    'dropout_rate': 0.7,  # Tăng dropout để giảm overfitting
    'num_beard_classes': 3,  # Có râu, không râu, không rõ
    'num_glasses_classes': 4,  # Kính thường, kính râm, không kính, không rõ
    
    # Training
    'batch_size': 16,  # Tăng batch size
    'epochs': 50,
    'learning_rate': 5e-5,  # Giảm learning rate
    'weight_decay': 1e-3,  # Tăng weight decay để tăng regularization
    'optimizer': 'adamw',  # Chuyển sang AdamW optimizer
    'scheduler': 'cosine',  # Chuyển sang cosine scheduler
    
    # Loss
    'loss_type': 'focal',  # ['cross_entropy', 'focal']
    'beard_weight': 1.0,
    'glasses_weight': 1.0,
    'focal_alpha': 0.25,  # Điều chỉnh focal loss alpha
    'focal_gamma': 2.0,
    
    # Other
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'num_workers': 0,
    'early_stopping_patience': 15,  # Tăng patience
    'grad_clip': 1.0,
    'image_size': 224
}

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main():
    # Tạo thư mục
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Lưu config
    config_path = Path(config['log_dir']) / 'head_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Khởi tạo trainer
    trainer = Trainer(config)
    
    # Train model
    best_val_acc = trainer.train()
    
    # Test model
    test_results = trainer.test_model()
    
    # Chuyển đổi kết quả test thành JSON serializable
    test_results = convert_numpy_types(test_results)
    
    # Lưu kết quả test
    results_path = Path(config['log_dir']) / 'head_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test results saved to {results_path}")

if __name__ == '__main__':
    main() 