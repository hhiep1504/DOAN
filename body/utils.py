import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from pathlib import Path

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Sử dụng để dừng training sớm nếu validation metric không cải thiện sau một số epoch nhất định.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print, mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Period after which we stop the training.
            verbose (bool): If True, prints a message for each validation metric improvement. 
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
            mode (str): 'min' or 'max'. In 'min' mode, training will stop when the quantity monitored has stopped decreasing; 
                        in 'max' mode it will stop when the quantity monitored has stopped increasing.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.mode = mode
        self.mode_str = 'decreased' if self.mode == 'min' else 'increased'
        
        if self.mode == 'min':
            self.best_score = np.Inf
        else: # mode == 'max'
             self.best_score = -np.Inf

    def __call__(self, current_score, model):
        """
        Args:
            current_score (float): Current value of the monitored metric on the validation set.
            model (torch.nn.Module): The model to save.
        """
        score = -current_score if self.mode == 'min' else current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(current_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(current_score, model)
            self.counter = 0

    def save_checkpoint(self, current_score, model):
        """Saves model when validation metric improves."""
        if self.verbose:
            self.trace_func(f'Validation metric {self.mode_str} ({self.best_score:.6f} --> {current_score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        # Cập nhật best_score sau khi lưu checkpoint
        self.best_score = -current_score if self.mode == 'min' else current_score


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Loss Function để giải quyết vấn đề class imbalance trong phân loại đa lớp.
    Công thức: -alpha * (1 - p_t)^gamma * log(p_t)
    Args:
        alpha (float or list): Weighting factor for each class or a single value.
                                Mặc định 0.25 trong paper gốc, có thể điều chỉnh.
        gamma (float): Focusing parameter. Gamma > 0 làm giảm trọng số của các ví dụ được phân loại tốt.
                       Thường là 2.0 trong paper gốc.
        reduction (str): 'mean', 'sum', 'none'. Cách giảm loss tensor thành scalar.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', num_classes=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.num_classes = num_classes

        if isinstance(self.alpha, (list, tuple)):
            # Convert alpha list to a tensor
            self.alpha = torch.Tensor(self.alpha)
            if num_classes and len(self.alpha) != num_classes:
                 raise ValueError(f"Length of alpha ({len(self.alpha)}) must match num_classes ({num_classes})")
        elif not isinstance(self.alpha, (int, float)):
            raise TypeError("Alpha must be an int, float, list or tuple.")
            
        # self.register_buffer('alpha_t', None) # Use buffer for device consistency

    def forward(self, inputs, targets):
        # inputs: raw logits [batch_size, num_classes]
        # targets: class indices [batch_size]
        
        # Ensure alpha is on the same device as inputs
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                 self.alpha = self.alpha.to(inputs.device)

        ce_loss = F.cross_entropy(inputs, targets, reduction='none') # Calculate CE loss per sample
        pt = torch.exp(-ce_loss) # Probability of the ground truth class

        # Calculate alpha_t based on target class
        if isinstance(self.alpha, torch.Tensor):
            # alpha_t = self.alpha_t[targets] if self.alpha_t is not None else self.alpha[targets]
             alpha_t = self.alpha[targets]
        else:
            alpha_t = self.alpha # Use scalar alpha

        # Focal Loss formula
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def plot_training_history(history, log_dir, title_suffix=""):
    """
    Plot training history (loss and accuracy).
    Vẽ biểu đồ lịch sử training (loss và accuracy).
    """
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Classification Accuracy {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = Path(log_dir) / f'training_history{title_suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    # plt.show() # Avoid showing plot directly in script

def plot_confusion_matrix(targets, preds, classes, log_dir, title_suffix=""):
    """
    Plot confusion matrix.
    Vẽ ma trận nhầm lẫn (confusion matrix).
    """
    cm = confusion_matrix(targets, preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix {title_suffix}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    save_path = Path(log_dir) / f'confusion_matrix{title_suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix plot saved to {save_path}")
    # plt.show() # Avoid showing plot directly in script 