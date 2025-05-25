import os
import json
import numpy as np
from pathlib import Path
from train import BodyTrainer

# Cấu hình training
config = {
    # Data
    'image_dir': 'data/body/hiep_dataset/body_images',
    'train_annotation': 'data/body/hiep_dataset/body_annotations/split_annotations/body_train.csv',
    'val_annotation': 'data/body/hiep_dataset/body_annotations/split_annotations/body_val.csv',
    'test_annotation': 'data/body/hiep_dataset/body_annotations/split_annotations/body_test.csv',
    'num_classes': 4, 
    # Model
    'backbone': 'resnet50',
    'pretrained': True,
    'freeze_backbone': False,
    'dropout_rate': 0.5,

    # Training
    'batch_size': 8,
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'optimizer': 'adamw',
    'scheduler': 'plateau',
    'scheduler_patience': 5,
    
    # Loss
    'loss_type': 'cross_entropy',
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    
    # Other
    'checkpoint_dir': 'checkpoints/body',
    'log_dir': 'logs/body',
    'num_workers': 0,
    'early_stopping_patience': 10,
    'early_stopping_delta': 0.001,
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
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def main():
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    config_path = Path(config['log_dir']) / 'body_config.json'
    json_serializable_config = convert_numpy_types(config)
    with open(config_path, 'w') as f:
        json.dump(json_serializable_config, f, indent=2)
    print(f"Configuration saved to {config_path}")

    trainer = BodyTrainer(config)
    
    best_val_acc = trainer.train()
    
    print(f"\n--- Training Process Finished ---")
    print(f"Best validation accuracy achieved: {best_val_acc:.4f}")

if __name__ == '__main__':
    main() 