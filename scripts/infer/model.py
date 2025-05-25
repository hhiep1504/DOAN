import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights  # Thêm dòng này


class FeatureClassifier(nn.Module):
    def __init__(self, freeze_backbone=True, dropout_rate=0.5):
        super(FeatureClassifier, self).__init__()
        
        # Load pretrained ResNet50
        # Thay đổi từ pretrained=True sang weights=ResNet50_Weights.IMAGENET1K_V1
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        
        # Freeze backbone parameters if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get input features dimension
        in_features = self.backbone.fc.in_features
        
        # Remove the original fc layer
        self.backbone.fc = nn.Identity()
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Tạo các classifier heads cho từng feature
        self.gender_classifier = nn.Linear(in_features, 3)      # 3 classes: Nam, Nữ, Unknown
        self.age_classifier = nn.Linear(in_features, 4)         # 9 age groups
        self.ethnicity_classifier = nn.Linear(in_features, 5)   # 5 ethnicity groups
        self.beard_classifier = nn.Linear(in_features, 3)       # 3 classes: Yes, No, Unknown
        self.glasses_classifier = nn.Linear(in_features, 4)     # 4 classes: Normal, Sun, No, Unknown
        self.accessories_classifier = nn.Linear(in_features, 8) # 0..6: types, 7:Unknown
        
    def extract_features(self, x):
        """Extract features using the backbone"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
        
    def forward(self, x):
        # Extract features
        features = self.extract_features(x)
        
        # Apply dropout
        features = self.dropout(features)
        
        # Apply classifiers
        return {
            'gender': self.gender_classifier(features),
            'age': self.age_classifier(features),
            'ethnicity': self.ethnicity_classifier(features),
            'beard': self.beard_classifier(features),
            'glasses': self.glasses_classifier(features),
            'accessories': self.accessories_classifier(features)
        }