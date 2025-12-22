"""
File: backend/models/outfit_encoder.py
Purpose: Extract visual features from outfit/accessory images
Author: Pabasara11
Date: 2025-12-18

Architecture:  ResNet-152 (Pre-trained on ImageNet)
Output:  2048-dimensional feature vectors

What it does:
1. Loads ResNet-152 backbone
2. Removes final classification layer
3. Adds custom accessory classification head
4. Fine-tunes on our accessory dataset
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import timm

class OutfitFeatureExtractor(nn.Module):
    """
    ResNet-152 based feature extractor for fashion items
    """
    
    def __init__(self, num_classes=15, pretrained=True, freeze_backbone=False):
        """
        Args: 
            num_classes (int): Number of accessory categories
            pretrained (bool): Use ImageNet pre-trained weights
            freeze_backbone (bool): Freeze ResNet layers during training
        """
        super(OutfitFeatureExtractor, self).__init__()
        
        print("🏗️  Building Outfit Feature Extractor (ResNet-152)...")
        
        # Load pre-trained ResNet-152
        self.backbone = models.resnet152(pretrained=pretrained)
        
        # Get feature dimension
        self.feature_dim = self.backbone.fc.in_features  # 2048 for ResNet-152
        
        # Remove original classification head
        self.backbone. fc = nn.Identity()
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("❄️  Backbone frozen (only training classifier)")
        else:
            print("🔥 Backbone unfrozen (full fine-tuning)")
        
        # Custom classification head for accessories
        self.classifier = nn. Sequential(
            nn. Dropout(0.5),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        print(f"✅ Model built!  Feature dimension: {self.feature_dim}")
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input images [batch_size, 3, 224, 224]
            
        Returns:
            features (torch.Tensor): Feature vectors [batch_size, 2048]
            logits (torch.Tensor): Classification logits [batch_size, num_classes]
        """
        # Extract features
        features = self.backbone(x)  # [batch_size, 2048]
        
        # Classification
        logits = self.classifier(features)  # [batch_size, num_classes]
        
        return features, logits
    
    def extract_features(self, x):
        """
        Extract only features (no classification)
        
        Args:
            x (torch. Tensor): Input images
            
        Returns:
            torch.Tensor: Feature vectors [batch_size, 2048]
        """
        with torch.no_grad():
            features = self.backbone(x)
        return features
    
    def get_transform(self):
        """
        Get image preprocessing transforms
        
        Returns: 
            torchvision.transforms.Compose: Transform pipeline
        """
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                std=[0.229, 0.224, 0.225]
            )
        ])


class VisionTransformerExtractor(nn.Module):
    """
    Alternative:  Vision Transformer (ViT) based extractor
    (Use if you want state-of-the-art performance)
    """
    
    def __init__(self, num_classes=15, model_name='vit_base_patch16_224', pretrained=True):
        """
        Args:
            num_classes (int): Number of accessory categories
            model_name (str): ViT model variant
            pretrained (bool): Use pre-trained weights
        """
        super(VisionTransformerExtractor, self).__init__()
        
        print(f"🏗️  Building Vision Transformer:  {model_name}...")
        
        # Load pre-trained ViT using timm
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features  # 768 for ViT-Base
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        print(f"✅ ViT built! Feature dimension: {self.feature_dim}")
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)  # [batch_size, 768]
        logits = self.classifier(features)
        return features, logits
    
    def extract_features(self, x):
        """Extract features only"""
        with torch.no_grad():
            features = self. backbone(x)
        return features
    
    def get_transform(self):
        """Image preprocessing"""
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])


def test_model():
    """
    Test function to verify model works
    """
    print("\n🧪 Testing Outfit Feature Extractor...")
    
    # Create model
    model = OutfitFeatureExtractor(num_classes=15, pretrained=False)
    
    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    features, logits = model(dummy_input)
    
    print(f"✅ Input shape: {dummy_input.shape}")
    print(f"✅ Features shape: {features.shape}")
    print(f"✅ Logits shape: {logits.shape}")
    
    assert features.shape == (batch_size, 2048), "Feature dimension mismatch!"
    assert logits.shape == (batch_size, 15), "Logits dimension mismatch!"
    
    print("✅ Model test passed!")


if __name__ == "__main__": 
    test_model()