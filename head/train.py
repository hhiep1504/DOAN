import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import time
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import create_data_loaders, HeadDataset, collate_fn
from model import create_model, MultiTaskLoss, freeze_backbone, count_parameters

class Trainer:
    """
    Trainer class for multi-task head features classification
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
            'train_beard_acc': [], 'val_beard_acc': [],
            'train_glasses_acc': [], 'val_glasses_acc': [],
            'train_combined_acc': [], 'val_combined_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
        self.early_stopping_counter = 0
        self.start_epoch = 0
    
    def _setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        # Định nghĩa transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),  # Tăng góc xoay
            transforms.RandomAffine(
                degrees=0,
                translate=(0.15, 0.15),  # Tăng độ dịch chuyển
                scale=(0.85, 1.15),     # Tăng độ co giãn
                shear=15               # Tăng độ nghiêng
            ),
            transforms.ColorJitter(
                brightness=0.3,    # Tăng độ sáng
                contrast=0.3,      # Tăng độ tương phản
                saturation=0.3,    # Tăng độ bão hòa
                hue=0.15          # Tăng độ thay đổi màu
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
        
        # Tạo datasets
        train_dataset = HeadDataset(
            image_dir=self.config['image_dir'],
            annotation_file=self.config['train_annotation'],
            transform=train_transform
        )
        
        val_dataset = HeadDataset(
            image_dir=self.config['image_dir'],
            annotation_file=self.config['val_annotation'],
            transform=val_transform
        )
        
        test_dataset = HeadDataset(
            image_dir=self.config['image_dir'],
            annotation_file=self.config['test_annotation'],
            transform=val_transform
        )
        
        # Tạo data loaders với batch size lớn hơn
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    def _setup_model(self):
        """Setup model"""
        print("Setting up model...")
        self.model = create_model(
            backbone=self.config['backbone'],
            num_beard_classes=self.config['num_beard_classes'],
            num_glasses_classes=self.config['num_glasses_classes'],
            pretrained=self.config['pretrained'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # Freeze backbone if specified
        if self.config.get('freeze_backbone', False):
            freeze_backbone(self.model, freeze=True)
        
        count_parameters(self.model)
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss function"""
        print("Setting up training components...")
        
        # Optimizer
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
        
        # Scheduler
        if self.config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        elif self.config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs']
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = MultiTaskLoss(
            beard_weight=self.config.get('beard_weight', 1.0),
            glasses_weight=self.config.get('glasses_weight', 1.0),
            loss_type=self.config.get('loss_type', 'cross_entropy'),
            focal_alpha=self.config.get('focal_alpha', 1.0),
            focal_gamma=self.config.get('focal_gamma', 2.0)
        )
    
    def calculate_accuracy(self, predictions, targets):
        """Calculate accuracy for multi-task predictions"""
        with torch.no_grad():
            beard_pred = torch.argmax(predictions['beard'], dim=1)
            glasses_pred = torch.argmax(predictions['glasses'], dim=1)
            
            beard_acc = (beard_pred == targets['beard']).float().mean().item()
            glasses_acc = (glasses_pred == targets['glasses']).float().mean().item()
            
            # Combined accuracy (both tasks correct)
            combined_acc = ((beard_pred == targets['beard']) & 
                           (glasses_pred == targets['glasses'])).float().mean().item()
            
            return beard_acc, glasses_acc, combined_acc
    
    def _validate_labels(self, targets):
        """Kiểm tra và xử lý nhãn không hợp lệ"""
        valid_beard = (targets['beard'] >= 0) & (targets['beard'] < self.config['num_beard_classes'])
        valid_glasses = (targets['glasses'] >= 0) & (targets['glasses'] < self.config['num_glasses_classes'])
        
        if not valid_beard.all():
            invalid_indices = torch.where(~valid_beard)[0]
            print(f"Warning: Found {len(invalid_indices)} invalid beard labels")
            print(f"Invalid values: {targets['beard'][invalid_indices]}")
            # Chuyển các nhãn không hợp lệ về 0
            targets['beard'][~valid_beard] = 0
            
        if not valid_glasses.all():
            invalid_indices = torch.where(~valid_glasses)[0]
            print(f"Warning: Found {len(invalid_indices)} invalid glasses labels")
            print(f"Invalid values: {targets['glasses'][invalid_indices]}")
            # Chuyển các nhãn không hợp lệ về 0
            targets['glasses'][~valid_glasses] = 0
            
        return targets

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_beard_acc = 0.0
        total_glasses_acc = 0.0
        total_combined_acc = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {
                'beard': batch['beard'].to(self.device).long(),
                'glasses': batch['glasses'].to(self.device).long()
            }
            
            # Kiểm tra và xử lý nhãn không hợp lệ
            targets = self._validate_labels(targets)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            
            # Calculate loss
            losses = self.criterion(predictions, targets)
            loss = losses['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Calculate accuracy
            beard_acc, glasses_acc, combined_acc = self.calculate_accuracy(predictions, targets)
            
            # Update metrics
            total_loss += loss.item()
            total_beard_acc += beard_acc
            total_glasses_acc += glasses_acc
            total_combined_acc += combined_acc
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'B_Acc': f'{beard_acc:.3f}',
                'G_Acc': f'{glasses_acc:.3f}',
                'C_Acc': f'{combined_acc:.3f}'
            })
        
        # Average metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_beard_acc = total_beard_acc / len(self.train_loader)
        avg_glasses_acc = total_glasses_acc / len(self.train_loader)
        avg_combined_acc = total_combined_acc / len(self.train_loader)
        
        return avg_loss, avg_beard_acc, avg_glasses_acc, avg_combined_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_beard_acc = 0.0
        total_glasses_acc = 0.0
        total_combined_acc = 0.0
        
        all_beard_preds = []
        all_glasses_preds = []
        all_beard_targets = []
        all_glasses_targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                targets = {
                    'beard': batch['beard'].to(self.device).long(),
                    'glasses': batch['glasses'].to(self.device).long()
                }
                
                # Kiểm tra và xử lý nhãn không hợp lệ
                targets = self._validate_labels(targets)
                
                # Forward pass
                predictions = self.model(images)
                
                # Calculate loss
                losses = self.criterion(predictions, targets)
                loss = losses['total_loss']
                
                # Calculate accuracy
                beard_acc, glasses_acc, combined_acc = self.calculate_accuracy(predictions, targets)
                
                # Update metrics
                total_loss += loss.item()
                total_beard_acc += beard_acc
                total_glasses_acc += glasses_acc
                total_combined_acc += combined_acc
                
                # Store predictions for detailed analysis
                beard_pred = torch.argmax(predictions['beard'], dim=1)
                glasses_pred = torch.argmax(predictions['glasses'], dim=1)
                
                all_beard_preds.extend(beard_pred.cpu().numpy())
                all_glasses_preds.extend(glasses_pred.cpu().numpy())
                all_beard_targets.extend(targets['beard'].cpu().numpy())
                all_glasses_targets.extend(targets['glasses'].cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'B_Acc': f'{beard_acc:.3f}',
                    'G_Acc': f'{glasses_acc:.3f}',
                    'C_Acc': f'{combined_acc:.3f}'
                })
        
        # Average metrics
        avg_loss = total_loss / len(self.val_loader)
        avg_beard_acc = total_beard_acc / len(self.val_loader)
        avg_glasses_acc = total_glasses_acc / len(self.val_loader)
        avg_combined_acc = total_combined_acc / len(self.val_loader)
        
        return (avg_loss, avg_beard_acc, avg_glasses_acc, avg_combined_acc,
                all_beard_preds, all_glasses_preds, all_beard_targets, all_glasses_targets)
    
    def test_model(self):
        """Test the model on test set"""
        print("Testing model...")
        self.model.eval()
        
        all_beard_preds = []
        all_glasses_preds = []
        all_beard_targets = []
        all_glasses_targets = []
        
        total_loss = 0.0
        total_beard_acc = 0.0
        total_glasses_acc = 0.0
        total_combined_acc = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            for batch in pbar:
                images = batch['image'].to(self.device)
                targets = {
                    'beard': batch['beard'].to(self.device),
                    'glasses': batch['glasses'].to(self.device)
                }
                
                predictions = self.model(images)
                losses = self.criterion(predictions, targets)
                loss = losses['total_loss']
                
                beard_acc, glasses_acc, combined_acc = self.calculate_accuracy(predictions, targets)
                
                total_loss += loss.item()
                total_beard_acc += beard_acc
                total_glasses_acc += glasses_acc
                total_combined_acc += combined_acc
                
                # Store predictions
                beard_pred = torch.argmax(predictions['beard'], dim=1)
                glasses_pred = torch.argmax(predictions['glasses'], dim=1)
                
                all_beard_preds.extend(beard_pred.cpu().numpy())
                all_glasses_preds.extend(glasses_pred.cpu().numpy())
                all_beard_targets.extend(targets['beard'].cpu().numpy())
                all_glasses_targets.extend(targets['glasses'].cpu().numpy())
        
        # Calculate final metrics
        avg_loss = total_loss / len(self.test_loader)
        avg_beard_acc = total_beard_acc / len(self.test_loader)
        avg_glasses_acc = total_glasses_acc / len(self.test_loader)
        avg_combined_acc = total_combined_acc / len(self.test_loader)
        
        # Print results
        print(f"\nTest Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Beard Accuracy: {avg_beard_acc:.4f}")
        print(f"Glasses Accuracy: {avg_glasses_acc:.4f}")
        print(f"Combined Accuracy: {avg_combined_acc:.4f}")
        
        # Generate classification reports
        beard_labels = ['No Beard', 'Beard'] if self.config['num_beard_classes'] == 2 else [f'Beard_{i}' for i in range(self.config['num_beard_classes'])]
        glasses_labels = ['No Glasses', 'Glasses'] if self.config['num_glasses_classes'] == 2 else [f'Glasses_{i}' for i in range(self.config['num_glasses_classes'])]
        
        print("\nBeard Classification Report:")
        print(classification_report(all_beard_targets, all_beard_preds, target_names=beard_labels))
        
        print("\nGlasses Classification Report:")
        print(classification_report(all_glasses_targets, all_glasses_preds, target_names=glasses_labels))
        
        return {
            'loss': avg_loss,
            'beard_acc': avg_beard_acc,
            'glasses_acc': avg_glasses_acc,
            'combined_acc': avg_combined_acc,
            'beard_preds': all_beard_preds,
            'glasses_preds': all_glasses_preds,
            'beard_targets': all_beard_targets,
            'glasses_targets': all_glasses_targets
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'history': self.history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'head_latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'head_best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation accuracy: {self.best_val_acc:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        
        print(f"Resumed from epoch {self.start_epoch}")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Beard accuracy plot
        axes[0, 1].plot(self.history['train_beard_acc'], label='Train Beard Acc')
        axes[0, 1].plot(self.history['val_beard_acc'], label='Val Beard Acc')
        axes[0, 1].set_title('Beard Classification Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Glasses accuracy plot
        axes[1, 0].plot(self.history['train_glasses_acc'], label='Train Glasses Acc')
        axes[1, 0].plot(self.history['val_glasses_acc'], label='Val Glasses Acc')
        axes[1, 0].set_title('Glasses Classification Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined accuracy plot
        axes[1, 1].plot(self.history['train_combined_acc'], label='Train Combined Acc')
        axes[1, 1].plot(self.history['val_combined_acc'], label='Val Combined Acc')
        axes[1, 1].set_title('Combined Task Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'head_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, beard_targets, beard_preds, glasses_targets, glasses_preds):
        """Plot confusion matrices for both tasks"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Beard confusion matrix
        beard_cm = confusion_matrix(beard_targets, beard_preds)
        beard_labels = ['No Beard', 'Beard'] if self.config['num_beard_classes'] == 2 else [f'B_{i}' for i in range(self.config['num_beard_classes'])]
        
        sns.heatmap(beard_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=beard_labels, yticklabels=beard_labels, ax=axes[0])
        axes[0].set_title('Beard Classification Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Glasses confusion matrix
        glasses_cm = confusion_matrix(glasses_targets, glasses_preds)
        glasses_labels = ['No Glasses', 'Glasses'] if self.config['num_glasses_classes'] == 2 else [f'G_{i}' for i in range(self.config['num_glasses_classes'])]
        
        sns.heatmap(glasses_cm, annot=True, fmt='d', cmap='Greens',
                   xticklabels=glasses_labels, yticklabels=glasses_labels, ax=axes[1])
        axes[1].set_title('Glasses Classification Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs...")
        print(f"Early stopping patience: {self.config.get('early_stopping_patience', 10)}")
        
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_beard_acc, train_glasses_acc, train_combined_acc = self.train_epoch()
            
            # Validation phase
            val_results = self.validate_epoch()
            val_loss, val_beard_acc, val_glasses_acc, val_combined_acc = val_results[:4]
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_beard_acc'].append(train_beard_acc)
            self.history['val_beard_acc'].append(val_beard_acc)
            self.history['train_glasses_acc'].append(train_glasses_acc)
            self.history['val_glasses_acc'].append(val_glasses_acc)
            self.history['train_combined_acc'].append(train_combined_acc)
            self.history['val_combined_acc'].append(val_combined_acc)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Beard Acc: {train_beard_acc:.4f} | Val Beard Acc: {val_beard_acc:.4f}")
            print(f"Train Glasses Acc: {train_glasses_acc:.4f} | Val Glasses Acc: {val_glasses_acc:.4f}")
            print(f"Train Combined Acc: {train_combined_acc:.4f} | Val Combined Acc: {val_combined_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Use combined accuracy for best model selection
            current_val_acc = val_combined_acc
            
            # Check for best model
            is_best = current_val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = current_val_acc
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(current_val_acc)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if self.early_stopping_counter >= self.config.get('early_stopping_patience', 10):
                print(f"\nEarly stopping triggered after {self.early_stopping_counter} epochs without improvement")
                break
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best validation combined accuracy: {self.best_val_acc:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        # Load best model for final evaluation
        best_checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pth'
        if best_checkpoint_path.exists():
            self.load_checkpoint(best_checkpoint_path)
            
            # Test on validation set with best model
            val_results = self.validate_epoch()
            val_loss, val_beard_acc, val_glasses_acc, val_combined_acc = val_results[:4]
            beard_preds, glasses_preds, beard_targets, glasses_targets = val_results[4:]
            
            print(f"\nBest model validation results:")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Beard Accuracy: {val_beard_acc:.4f}")
            print(f"Glasses Accuracy: {val_glasses_acc:.4f}")
            print(f"Combined Accuracy: {val_combined_acc:.4f}")
            
            # Plot confusion matrices
            self.plot_confusion_matrices(beard_targets, beard_preds, glasses_targets, glasses_preds)
        
        # Save final history
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        
        return self.best_val_acc