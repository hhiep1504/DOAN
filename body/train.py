import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import time
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import create_body_data_loaders, BodyDataset, body_collate_fn
from model import create_model
from utils import EarlyStopping, FocalLoss, plot_training_history, plot_confusion_matrix
from model import freeze_backbone, count_parameters

class BodyTrainer:
    """
    Trainer class for body accessories classification
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.log_dir = Path(config['log_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }
        
        self.start_epoch = 0
        
        # Initialize EarlyStopping
        self.early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 10),
            verbose=True,
            delta=self.config.get('early_stopping_delta', 0),
            path=self.checkpoint_dir / 'body_best_checkpoint.pth',
            mode='max' 
        )
    
    def _setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        train_transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),  
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),  
                scale=(0.85, 1.15),     
                shear=15               
            ),
            transforms.ColorJitter(
                brightness=0.3,    
                contrast=0.3,      
                saturation=0.3,    
                hue=0.15          
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((self.config['image_size'], self.config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.train_dataset = BodyDataset(
            image_dir=self.config['image_dir'],
            annotation_file=self.config['train_annotation'],
            transform=train_transform,
        )
        
        self.val_dataset = BodyDataset(
            image_dir=self.config['image_dir'],
            annotation_file=self.config['val_annotation'],
            transform=val_transform,
        )
        
        self.test_dataset = BodyDataset(
            image_dir=self.config['image_dir'],
            annotation_file=self.config['test_annotation'],
            transform=val_transform,
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=body_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=body_collate_fn
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=body_collate_fn
        )
    
    def _setup_model(self):
        """Setup model"""
        print("Setting up model...")
        self.model = create_model(
            backbone=self.config['backbone'],
            num_classes=self.config['num_classes'],
            pretrained=self.config['pretrained'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        if self.config.get('freeze_backbone', False):
            freeze_backbone(self.model, freeze=True)
        
        count_parameters(self.model)
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        print("Setting up training components...")
        
        if self.config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
             raise ValueError(f"Optimizer {self.config['optimizer']} không được hỗ trợ")
        
        if self.config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=self.config.get('scheduler_patience', 5), verbose=True
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs']
            )
        elif self.config['scheduler'] is None or self.config['scheduler'] == 'none':
            self.scheduler = None
        else:
             raise ValueError(f"Scheduler {self.config['scheduler']} không được hỗ trợ")
        
        if self.config.get('loss_type', 'cross_entropy') == 'focal':
             self.criterion = FocalLoss(
                 alpha=self.config.get('focal_alpha', 0.25),
                 gamma=self.config.get('focal_gamma', 2.0),
                 num_classes=self.config['num_classes']
             )
        else:
            self.criterion = nn.CrossEntropyLoss() 
            
    def calculate_accuracy(self, predictions, targets):
        """Calculate accuracy for accessories classification"""
        with torch.no_grad():
            predicted_classes = torch.argmax(predictions, dim=1)
            correct_predictions = (predicted_classes == targets).sum().item()
            total_samples = targets.size(0)
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            return accuracy
    
    def _validate_labels(self, targets):
        """Kiểm tra và xử lý nhãn không hợp lệ"""
        num_classes = self.config['num_classes']
        valid_labels = (targets >= 0) & (targets < num_classes)
        
        if not valid_labels.all():
            invalid_indices = torch.where(~valid_labels)[0]
            print(f"Warning: Found {len(invalid_indices)} invalid labels")
            print(f"Invalid values: {targets[invalid_indices]}")
            targets[~valid_labels] = 0 
            
        return targets

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['accessories'].to(self.device).long()
            
            targets = self._validate_labels(targets)
            
            self.optimizer.zero_grad()
            predictions = self.model(images) 
            
            loss = self.criterion(predictions, targets)
            
            loss.backward()
            
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            acc = self.calculate_accuracy(predictions, targets)
            
            total_loss += loss.item()
            total_acc += acc
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.3f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_acc / len(self.train_loader)
        
        return avg_loss, avg_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['accessories'].to(self.device).long()
                
                targets = self._validate_labels(targets)
                
                predictions = self.model(images) 
                
                loss = self.criterion(predictions, targets)
                acc = self.calculate_accuracy(predictions, targets)
                
                total_loss += loss.item()
                total_acc += acc
                
                predicted_classes = torch.argmax(predictions, dim=1)
                all_preds.extend(predicted_classes.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{acc:.3f}'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_acc / len(self.val_loader)
        
        return (avg_loss, avg_acc, all_preds, all_targets)
    
    def test_model(self):
        """Test the model on test set"""
        print("Testing model...")
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        total_loss = 0.0
        total_acc = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = batch['accessories'].to(self.device).long()
                
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
                
                acc = self.calculate_accuracy(predictions, targets)
                
                total_loss += loss.item()
                total_acc += acc
                
                predicted_classes = torch.argmax(predictions, dim=1)
                all_preds.extend(predicted_classes.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        avg_acc = total_acc / len(self.test_loader)
        
        print(f"\nTest Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {avg_acc:.4f}")
        
        accessories_labels = list(self.config.get('accessories_mapping', {}).keys())
        if not accessories_labels:
             accessories_labels = [f'Class_{i}' for i in range(self.config['num_classes'])]
             print("Warning: accessories_mapping not found in config. Using generic labels for report.")

        print("\nAccessories Classification Report:")
        print(classification_report(all_targets, all_preds, target_names=accessories_labels))
        
        return {
            'loss': avg_loss,
            'acc': avg_acc,
            'preds': all_preds,
            'targets': all_targets,
            'classes': accessories_labels
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.early_stopping.best_score if self.early_stopping.mode == 'max' else -self.early_stopping.best_score,
            'config': self.config,
            'history': self.history,
            'early_stopping_counter': self.early_stopping.counter
        }
        
        checkpoint_path = self.checkpoint_dir / 'body_latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        
        if not Path(checkpoint_path).exists():
             print(f"Error: Checkpoint file not found at {checkpoint_path}")
             return False 

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        
        self.history = checkpoint.get('history', self.history)
        
        if 'best_val_acc' in checkpoint:
            self.early_stopping.best_score = checkpoint['best_val_acc'] if self.early_stopping.mode == 'max' else -checkpoint['best_val_acc']
        self.early_stopping.counter = checkpoint.get('early_stopping_counter', 0)
        
        print(f"Resumed from epoch {self.start_epoch}")
        return True
    
    def plot_results(self, val_preds, val_targets, test_results):
        """Plot training history and confusion matrices"""
        plot_training_history(self.history, self.log_dir, title_suffix="_body")
        
        val_labels = list(self.config.get('accessories_mapping', {}).keys())
        if not val_labels:
             val_labels = [f'Class_{i}' for i in range(self.config['num_classes'])]
             print("Warning: accessories_mapping not found in config. Using generic labels for validation plot.")

        plot_confusion_matrix(val_targets, val_preds, val_labels, self.log_dir, title_suffix="_val_body")

        plot_confusion_matrix(test_results['targets'], test_results['preds'], test_results['classes'], self.log_dir, title_suffix="_test_body")
        
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        
        start_time = time.time()
        
        latest_checkpoint_path = self.checkpoint_dir / 'body_latest_checkpoint.pth'
        if latest_checkpoint_path.exists():
             print("Found latest checkpoint, attempting to resume...")
             self.load_checkpoint(latest_checkpoint_path)
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            train_loss, train_acc = self.train_epoch()
            
            val_results = self.validate_epoch()
            val_loss, val_acc, val_preds, val_targets = val_results
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            self.early_stopping(val_acc, self.model)
            
            self.save_checkpoint(epoch)
            
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            if self.early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation accuracy: {self.early_stopping.best_score:.4f}")
        
        best_checkpoint_path = self.checkpoint_dir / 'body_best_checkpoint.pth'
        if best_checkpoint_path.exists():
            print(f"\nLoading best model from {best_checkpoint_path} for final evaluation...")
            
            best_model = create_model(
                 backbone=self.config['backbone'],
                 num_classes=self.config['num_classes'], 
                 pretrained=False,
                 dropout_rate=self.config['dropout_rate']
             ).to(self.device)
            best_model.load_state_dict(torch.load(best_checkpoint_path, map_location=self.device))
            
            original_model = self.model
            self.model = best_model

            test_results = self.test_model()
            
            self.model = original_model

            print("\nEvaluating best model on validation set for plotting...")
            val_results_best_model = self.validate_epoch()
            val_loss_best, val_acc_best, val_preds_best, val_targets_best = val_results_best_model
            self.model = original_model

            print(f"\nBest model validation results (for plotting):")
            print(f"Validation Loss: {val_loss_best:.4f}")
            print(f"Validation Accuracy: {val_acc_best:.4f}")

            self.plot_results(val_preds_best, val_targets_best, test_results)

        else:
             print(f"\nWarning: Best checkpoint not found at {best_checkpoint_path}. Skipping final evaluation with best model.")
        
        history_path = self.log_dir / 'body_training_history.json' 
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        
        return self.early_stopping.best_score

if __name__ == "__main__":
    config = {
        'image_size': 224,
        'num_classes': 4,
        'backbone': 'resnet50',
        'pretrained': True,
        'dropout_rate': 0.5,
        
        'image_dir': 'data/body/hiep_dataset/body_images',
        'train_annotation': 'data/body/hiep_dataset/body_annotations/body_train.csv',
        'val_annotation': 'data/body/hiep_dataset/body_annotations/body_val.csv',
        'test_annotation': 'data/body/hiep_dataset/body_annotations/body_test.csv',
        'checkpoint_dir': 'checkpoints/body',
        'log_dir': 'logs/body',
        'accessories_mapping': {'Nothing': 0, 'Bag': 1, 'Backpack': 2, 'Other': 3},
        
        'epochs': 50,
        'batch_size': 64,
        'learning_rate': 0.001,
        'optimizer': 'adamw',
        'scheduler': 'plateau',
        'scheduler_patience': 5,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        'num_workers': 4,
        'early_stopping_patience': 10,
        'early_stopping_delta': 0.001,
        
        'loss_type': 'cross_entropy',
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
    }
    
    trainer = BodyTrainer(config)
    trainer.train() 