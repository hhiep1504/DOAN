import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, MobileNet_V3_Large_Weights

class MultiTaskHeadClassifier(nn.Module):
    """
    Multi-task classifier for beard and glasses detection
    """
    def __init__(self, backbone='resnet50', num_beard_classes=3, num_glasses_classes=4, 
                 pretrained=True, dropout_rate=0.5):
        """
        Args:
            backbone: Backbone architecture ('resnet50', 'efficientnet_b0', 'mobilenet_v3')
            num_beard_classes: Number of beard classes (3: có râu, không râu, không rõ)
            num_glasses_classes: Number of glasses classes (4: kính thường, kính râm, không kính, không rõ)
            pretrained: Use pretrained weights
            dropout_rate: Dropout rate
        """
        super(MultiTaskHeadClassifier, self).__init__()
        
        self.backbone_name = backbone
        self.num_beard_classes = num_beard_classes
        self.num_glasses_classes = num_glasses_classes
        
        # Initialize backbone
        self.backbone, self.feature_dim = self._create_backbone(backbone, pretrained)
        
        # Shared feature extractor
        self.shared_features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate)
        )
        
        # Task-specific heads
        self.beard_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_beard_classes)
        )
        
        self.glasses_classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_glasses_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_backbone(self, backbone, pretrained):
        """Create backbone network"""
        if backbone == 'resnet50':
            if pretrained:
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                model = models.resnet50(weights=None)
            # Remove final FC layer
            backbone_model = nn.Sequential(*list(model.children())[:-2])
            feature_dim = 2048
            
        elif backbone == 'efficientnet_b0':
            if pretrained:
                model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                model = models.efficientnet_b0(weights=None)
            # Remove final classifier
            backbone_model = model.features
            feature_dim = 1280
            
        elif backbone == 'mobilenet_v3':
            if pretrained:
                model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            else:
                model = models.mobilenet_v3_large(weights=None)
            # Remove final classifier
            backbone_model = model.features
            feature_dim = 960
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return backbone_model, feature_dim
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for module in [self.beard_classifier, self.glasses_classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 3, H, W]
            
        Returns:
            Dictionary with beard and glasses logits
        """
        # Extract features
        features = self.backbone(x)
        shared_features = self.shared_features(features)
        
        # Task-specific predictions
        beard_logits = self.beard_classifier(shared_features)
        glasses_logits = self.glasses_classifier(shared_features)
        
        return {
            'beard': beard_logits,
            'glasses': glasses_logits
        }
    
    def get_feature_extractor(self):
        """Get feature extractor for transfer learning or feature extraction"""
        return nn.Sequential(
            self.backbone,
            self.shared_features
        )

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining beard and glasses classification losses
    """
    def __init__(self, beard_weight=1.0, glasses_weight=1.0, loss_type='cross_entropy', 
                 focal_alpha=1.0, focal_gamma=2.0):
        """
        Args:
            beard_weight: Weight for beard classification loss
            glasses_weight: Weight for glasses classification loss
            loss_type: 'cross_entropy' or 'focal'
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super(MultiTaskLoss, self).__init__()
        self.beard_weight = beard_weight
        self.glasses_weight = glasses_weight
        
        if loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predictions, targets):
        """
        Calculate multi-task loss
        
        Args:
            predictions: Dictionary with 'beard' and 'glasses' logits
            targets: Dictionary with 'beard' and 'glasses' labels
            
        Returns:
            Total loss and individual losses
        """
        beard_loss = self.criterion(predictions['beard'], targets['beard'])
        glasses_loss = self.criterion(predictions['glasses'], targets['glasses'])
        
        total_loss = (self.beard_weight * beard_loss + 
                     self.glasses_weight * glasses_loss)
        
        return {
            'total_loss': total_loss,
            'beard_loss': beard_loss,
            'glasses_loss': glasses_loss
        }

def create_model(backbone='resnet50', num_beard_classes=3, num_glasses_classes=4,
                pretrained=True, dropout_rate=0.5):
    """
    Create multi-task head classifier model
    
    Args:
        backbone: Backbone architecture
        num_beard_classes: Number of beard classes
        num_glasses_classes: Number of glasses classes
        pretrained: Use pretrained weights
        dropout_rate: Dropout rate
        
    Returns:
        Model instance
    """
    model = MultiTaskHeadClassifier(
        backbone=backbone,
        num_beard_classes=num_beard_classes,
        num_glasses_classes=num_glasses_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    return model

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

def freeze_backbone(model, freeze=True):
    """
    Freeze/unfreeze backbone parameters
    
    Args:
        model: Model instance
        freeze: Whether to freeze backbone
    """
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    print(f"Backbone parameters {status}")

if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test different backbones
    backbones = ['resnet50', 'efficientnet_b0', 'mobilenet_v3']
    
    for backbone in backbones:
        print(f"\n--- Testing {backbone} ---")
        model = create_model(backbone=backbone)
        count_parameters(model)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Beard output shape: {output['beard'].shape}")
            print(f"Glasses output shape: {output['glasses'].shape}")
    
    # Test loss function
    print("\n--- Testing loss function ---")
    loss_fn = MultiTaskLoss(loss_type='focal')
    
    # Dummy predictions and targets
    predictions = {
        'beard': torch.randn(4, 3),
        'glasses': torch.randn(4, 4)
    }
    targets = {
        'beard': torch.randint(0, 3, (4,)),
        'glasses': torch.randint(0, 4, (4,))
    }
    
    losses = loss_fn(predictions, targets)
    print(f"Total loss: {losses['total_loss']:.4f}")
    print(f"Beard loss: {losses['beard_loss']:.4f}")
    print(f"Glasses loss: {losses['glasses_loss']:.4f}")
    
    print("\nModel testing completed!")