import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights, MobileNet_V3_Large_Weights

class BodyClassifier(nn.Module):
    """
    Classifier cho việc phân loại accessories (Bag, Backpack, Other, Nothing)
    """
    def __init__(self, backbone='resnet50', num_classes=4, pretrained=True, dropout_rate=0.5):
        """
        Args:
            backbone: Backbone architecture ('resnet50', 'efficientnet_b0', 'mobilenet_v3')
            num_classes: Số lượng lớp accessories (4: Bag, Backpack, Other, Nothing)
            pretrained: Sử dụng pretrained weights
            dropout_rate: Tỷ lệ dropout
        """
        super(BodyClassifier, self).__init__()
        
        self.backbone_name = backbone
        
        # Khởi tạo backbone
        self.backbone, self.feature_dim = self._create_backbone(backbone, pretrained)
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Khởi tạo weights
        self._initialize_weights()
    
    def _create_backbone(self, backbone, pretrained):
        """Tạo backbone network"""
        if backbone == 'resnet50':
            if pretrained:
                model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            else:
                model = models.resnet50(weights=None)
            backbone_model = nn.Sequential(*list(model.children())[:-2])
            feature_dim = 2048
            
        elif backbone == 'efficientnet_b0':
            if pretrained:
                model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            else:
                model = models.efficientnet_b0(weights=None)
            backbone_model = model.features
            feature_dim = 1280
            
        elif backbone == 'mobilenet_v3':
            if pretrained:
                model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            else:
                model = models.mobilenet_v3_large(weights=None)
            backbone_model = model.features
            feature_dim = 960
            
        else:
            raise ValueError(f"Backbone không được hỗ trợ: {backbone}")
        
        return backbone_model, feature_dim
    
    def _initialize_weights(self):
        """Khởi tạo weights cho classifier"""
        for m in self.classifier.modules():
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
            Logits cho việc phân loại accessories
        """
        features = self.backbone(x)
        features = self.feature_extractor(features)
        logits = self.classifier(features)
        return logits
    
    def get_feature_extractor(self):
        """Lấy feature extractor cho transfer learning hoặc feature extraction"""
        return nn.Sequential(
            self.backbone,
            self.feature_extractor
        )

class FocalLoss(nn.Module):
    """
    Focal Loss để giải quyết vấn đề class imbalance
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

def create_model(backbone='resnet50', num_classes=4, pretrained=True, dropout_rate=0.5):
    """
    Tạo model classifier cho accessories
    
    Args:
        backbone: Backbone architecture
        num_classes: Số lượng lớp accessories
        pretrained: Sử dụng pretrained weights
        dropout_rate: Tỷ lệ dropout
        
    Returns:
        Model instance
    """
    model = BodyClassifier(
        backbone=backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )
    return model

def count_parameters(model):
    """Đếm tổng số parameters và trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Tổng số parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

def freeze_backbone(model, freeze=True):
    """
    Freeze/unfreeze backbone parameters
    
    Args:
        model: Model instance
        freeze: Có freeze backbone hay không
    """
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    
    status = "đã freeze" if freeze else "đã unfreeze"
    print(f"Backbone parameters {status}")

if __name__ == "__main__":
    # Test model creation
    print("Đang test model creation...")
    
    # Test các backbone khác nhau
    backbones = ['resnet50', 'efficientnet_b0', 'mobilenet_v3']
    
    for backbone in backbones:
        print(f"\n--- Testing {backbone} ---")
        model = create_model(backbone=backbone)
        count_parameters(model)
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
    
    # Test loss function
    print("\n--- Testing loss function ---")
    loss_fn = FocalLoss()
    
    # Dummy predictions và targets
    predictions = torch.randn(4, 4)  # 4 samples, 4 classes
    targets = torch.randint(0, 4, (4,))
    
    loss = loss_fn(predictions, targets)
    print(f"Loss: {loss:.4f}")
    
    print("\nModel testing completed!")
