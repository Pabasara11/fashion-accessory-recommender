"""
File: backend/core/train_outfit_encoder.py
Purpose: Train the outfit feature extractor
Author: Pabasara11
Date: 2025-12-18

Training Strategy:
1. Load pre-trained ResNet-152
2. Freeze backbone, train classifier (5 epochs)
3. Unfreeze all, fine-tune (15 epochs)
4. Save best model checkpoint
"""

import torch
import torch. nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import sys

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))
from models.outfit_encoder import OutfitFeatureExtractor


class AccessoryDataset(Dataset):
    """
    Custom Dataset for accessory images
    """
    
    def __init__(self, data_dir, metadata_csv, transform=None):
        """
        Args:
            data_dir (str): Directory with train/val folders
            metadata_csv (str): Path to metadata CSV
            transform:  Image transformations
        """
        self. data_dir = Path(data_dir)
        self.metadata = pd.read_csv(metadata_csv)
        self.transform = transform
        
        # Create category to index mapping
        self.categories = sorted(self.metadata['articleType']. unique())
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx. items()}
        
        print(f"📊 Dataset loaded: {len(self.metadata)} images")
        print(f"📂 Categories ({len(self.categories)}): {self.categories[: 5]}...")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        """
        Get image and label
        """
        row = self.metadata. iloc[idx]
        
        # Image path
        img_id = row['id']
        category = row['articleType']. replace('/', '_')
        
        # Try to find image
        img_path = None
        for split in ['train', 'val']: 
            candidate_path = self.data_dir / split / category / f"{img_id}.jpg"
            if candidate_path. exists():
                img_path = candidate_path
                break
        
        if img_path is None or not img_path.exists():
            # Return black image if not found
            image = Image.new('RGB', (224, 224), color='black')
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except: 
                image = Image.new('RGB', (224, 224), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.category_to_idx[row['articleType']]
        
        return image, label


class OutfitEncoderTrainer:
    """
    Trainer class for outfit feature extractor
    """
    
    def __init__(self, data_dir, output_dir, device='cuda'):
        """
        Args:
            data_dir (str): Path to processed data
            output_dir (str): Path to save checkpoints
            device (str): 'cuda' or 'cpu'
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch. device(device if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Using device: {self.device}")
        
        # Load datasets
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.num_classes = 0
        
        self._setup_data()
        
        # Model
        self. model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
        # Training stats
        self.train_losses = []
        self. val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
    
    def _setup_data(self):
        """Setup datasets and dataloaders"""
        print("\n📊 Setting up datasets...")
        
        # Get transforms
        model = OutfitFeatureExtractor(num_classes=1, pretrained=False)
        transform = model.get_transform()
        
        # Load datasets
        train_csv = self.data_dir / 'train_metadata.csv'
        val_csv = self.data_dir / 'val_metadata.csv'
        
        self.train_dataset = AccessoryDataset(
            self.data_dir,
            train_csv,
            transform=transform
        )
        
        self.val_dataset = AccessoryDataset(
            self.data_dir,
            val_csv,
            transform=transform
        )
        
        self.num_classes = len(self.train_dataset.categories)
        
        # Create dataloaders
        self. train_loader = DataLoader(
            self.train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"✅ Train batches: {len(self.train_loader)}")
        print(f"✅ Val batches: {len(self.val_loader)}")
    
    def build_model(self, freeze_backbone=False):
        """Build and initialize model"""
        print(f"\n🏗️  Building model (freeze_backbone={freeze_backbone})...")
        
        self.model = OutfitFeatureExtractor(
            num_classes=self.num_classes,
            pretrained=True,
            freeze_backbone=freeze_backbone
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.001
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler. ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        print("✅ Model ready!")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            features, logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss. item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate model"""
        self.model. eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self. val_loader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                features, logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100 * correct / total:.2f}%"
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs=20, phase1_epochs=5):
        """
        Full training pipeline
        
        Args:
            num_epochs (int): Total epochs
            phase1_epochs (int): Epochs for phase 1 (frozen backbone)
        """
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        
        # PHASE 1: Train classifier only
        print("\n📍 PHASE 1: Training classifier (backbone frozen)")
        self.build_model(freeze_backbone=True)
        
        for epoch in range(phase1_epochs):
            print(f"\nEpoch {epoch+1}/{phase1_epochs}")
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:  {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save stats
            self.train_losses. append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
        
        # PHASE 2: Fine-tune entire model
        print("\n📍 PHASE 2: Fine-tuning entire model")
        self.build_model(freeze_backbone=False)
        
        for epoch in range(num_epochs - phase1_epochs):
            print(f"\nEpoch {phase1_epochs + epoch + 1}/{num_epochs}")
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:. 2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:. 2f}%")
            
            # Save stats
            self.train_losses. append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pth', epoch, val_acc)
                print(f"💾 Saved new best model! (Val Acc: {val_acc:. 2f}%)")
        
        # Save final model
        self.save_checkpoint('final_model.pth', num_epochs, val_acc)
        
        # Save training history
        self.save_training_history()
        
        print("\n" + "="*60)
        print("✅ TRAINING COMPLETE!")
        print(f"🏆 Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print("="*60)
    
    def save_checkpoint(self, filename, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self. optimizer.state_dict(),
            'val_acc': val_acc,
            'num_classes': self.num_classes,
            'category_mapping': self.train_dataset.category_to_idx
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved:  {path}")
    
    def save_training_history(self):
        """Save training statistics"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self. val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc
        }
        
        path = self.output_dir / 'training_history.json'
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"📊 Training history saved: {path}")


def main():
    """
    Main training function
    """
    # Paths (ADJUST THESE)
    DATA_DIR = r"E:\Research\AI_Projects\fashion-accessory-recommender\data\processed\accessories"
    OUTPUT_DIR = r"E:\Research\AI_Projects\fashion-accessory-recommender\backend\models\checkpoints"
    
    # Create trainer
    trainer = OutfitEncoderTrainer(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        device='cuda'  # Change to 'cpu' if no GPU
    )
    
    # Train
    trainer.train(num_epochs=20, phase1_epochs=5)


if __name__ == "__main__":
    main()