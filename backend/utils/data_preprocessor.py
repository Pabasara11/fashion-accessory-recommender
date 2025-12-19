"""
File: backend/utils/data_preprocessor. py
Purpose: Preprocess Kaggle fashion dataset
Author:  Pabasara11
Date: 2025-12-18

What it does:
1. Loads styles.csv metadata
2. Filters for accessories
3. Organizes images by category
4. Creates train/validation splits
5. Generates data statistics
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

class FashionDataPreprocessor: 
    """
    Preprocesses the Kaggle Fashion Product Images dataset
    """
    
    def __init__(self, raw_data_path, processed_data_path):
        """
        Args:
            raw_data_path (str): Path to raw dataset folder
            processed_data_path (str): Path to save processed data
        """
        self. raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.styles_csv = self.raw_data_path / "styles.csv"
        self.images_folder = self.raw_data_path / "images"
        
        # Create output directories
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
    def load_metadata(self):
        """
        Load and clean the styles.csv file
        
        Returns:
            pd.DataFrame: Cleaned metadata
        """
        print("📊 Loading metadata from styles.csv...")
        
        # Load CSV
        df = pd.read_csv(self.styles_csv, on_bad_lines='skip')
        
        print(f"✅ Loaded {len(df)} products")
        print(f"📋 Columns: {df.columns.tolist()}")
        
        # Clean data
        df = df.dropna(subset=['id', 'articleType'])
        
        print(f"✅ After cleaning: {len(df)} products")
        
        return df
    
    def filter_accessories(self, df):
        """
        Filter dataset to only include accessories
        
        Args:
            df (pd.DataFrame): Full dataset
            
        Returns: 
            pd.DataFrame: Accessories only
        """
        print("\n🔍 Filtering for accessories...")
        
        # Define accessory categories
        accessory_types = [
            'Watches', 'Watch', 'Sunglasses', 'Sunglass',
            'Bags', 'Handbag', 'Backpack', 'Clutch',
            'Jewellery', 'Bracelet', 'Necklace', 'Earrings', 'Ring',
            'Belts', 'Belt',
            'Scarves', 'Scarf', 'Stole', 'Muffler',
            'Caps', 'Hat', 'Cap',
            'Wallets', 'Wallet'
        ]
        
        # Filter
        accessories_df = df[df['articleType'].isin(accessory_types)]
        
        print(f"✅ Found {len(accessories_df)} accessories")
        
        # Show distribution
        print("\n📊 Accessory Distribution:")
        print(accessories_df['articleType'].value_counts())
        
        return accessories_df
    
    def organize_images(self, df, split_ratio=0.8):
        """
        Organize images into train/val folders by category
        
        Args: 
            df (pd.DataFrame): Accessories dataframe
            split_ratio (float): Train/validation split ratio
        """
        print("\n📁 Organizing images into train/val splits...")
        
        # Create directories
        train_dir = self.processed_data_path / 'train'
        val_dir = self.processed_data_path / 'val'
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Split data
        train_df, val_df = train_test_split(
            df, 
            test_size=1-split_ratio, 
            stratify=df['articleType'],
            random_state=42
        )
        
        print(f"✅ Train set: {len(train_df)} images")
        print(f"✅ Validation set: {len(val_df)} images")
        
        # Copy files
        self._copy_images(train_df, train_dir, "Training")
        self._copy_images(val_df, val_dir, "Validation")
        
        # Save metadata
        train_df.to_csv(self.processed_data_path / 'train_metadata.csv', index=False)
        val_df.to_csv(self.processed_data_path / 'val_metadata. csv', index=False)
        
        print("\n✅ Metadata saved!")
        
    def _copy_images(self, df, dest_dir, split_name):
        """
        Helper function to copy images to destination
        
        Args:
            df (pd.DataFrame): Dataframe with image IDs
            dest_dir (Path): Destination directory
            split_name (str): Name for progress bar
        """
        copied = 0
        missing = 0
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name} images"):
            # Source image path
            img_id = row['id']
            src_path = self.images_folder / f"{img_id}.jpg"
            
            # Create category subfolder
            category = row['articleType']. replace('/', '_')
            category_dir = dest_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Destination path
            dest_path = category_dir / f"{img_id}.jpg"
            
            # Copy if exists
            if src_path.exists():
                shutil.copy2(src_path, dest_path)
                copied += 1
            else:
                missing += 1
        
        print(f"  ✅ Copied:  {copied} images")
        if missing > 0:
            print(f"  ⚠️  Missing: {missing} images")
    
    def generate_statistics(self):
        """
        Generate dataset statistics
        """
        print("\n📊 Generating statistics...")
        
        stats = {
            'total_images': 0,
            'train_images': 0,
            'val_images': 0,
            'categories': {},
            'color_distribution': {},
            'season_distribution': {}
        }
        
        # Count images
        train_dir = self.processed_data_path / 'train'
        val_dir = self.processed_data_path / 'val'
        
        if train_dir.exists():
            stats['train_images'] = sum(1 for _ in train_dir.rglob('*.jpg'))
        
        if val_dir.exists():
            stats['val_images'] = sum(1 for _ in val_dir.rglob('*.jpg'))
        
        stats['total_images'] = stats['train_images'] + stats['val_images']
        
        # Load metadata for additional stats
        if (self.processed_data_path / 'train_metadata.csv').exists():
            train_meta = pd.read_csv(self. processed_data_path / 'train_metadata.csv')
            
            # Category distribution
            stats['categories'] = train_meta['articleType'].value_counts().to_dict()
            
            # Color distribution (if available)
            if 'baseColour' in train_meta.columns:
                stats['color_distribution'] = train_meta['baseColour'].value_counts().head(10).to_dict()
            
            # Season distribution (if available)
            if 'season' in train_meta.columns:
                stats['season_distribution'] = train_meta['season'].value_counts().to_dict()
        
        # Save statistics
        stats_path = self.processed_data_path / 'dataset_statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"✅ Statistics saved to:  {stats_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("📊 DATASET SUMMARY")
        print("="*50)
        print(f"Total Images: {stats['total_images']}")
        print(f"Train Images: {stats['train_images']}")
        print(f"Validation Images: {stats['val_images']}")
        print(f"\nCategories: {len(stats['categories'])}")
        for cat, count in list(stats['categories'].items())[:5]:
            print(f"  - {cat}: {count}")
        print("="*50)
        
        return stats
    
    def run_full_pipeline(self):
        """
        Run complete preprocessing pipeline
        """
        print("🚀 Starting Full Preprocessing Pipeline...")
        print("="*60)
        
        # Step 1: Load metadata
        df = self.load_metadata()
        
        # Step 2: Filter accessories
        accessories_df = self.filter_accessories(df)
        
        # Step 3: Organize images
        self.organize_images(accessories_df)
        
        # Step 4: Generate statistics
        stats = self.generate_statistics()
        
        print("\n✅ PREPROCESSING COMPLETE!")
        print("="*60)
        
        return stats


def main():
    """
    Main execution function
    """
    # Define paths (ADJUST THESE TO YOUR SYSTEM)
    RAW_DATA_PATH = r"E:\Research\AI_Projects\fashion-accessory-recommender\data\raw\fashion-dataset"
    PROCESSED_DATA_PATH = r"E:\Research\AI_Projects\fashion-accessory-recommender\data\processed\accessories"
    
    print("🎯 Fashion Dataset Preprocessor")
    print(f"📂 Raw Data: {RAW_DATA_PATH}")
    print(f"📂 Output: {PROCESSED_DATA_PATH}")
    print()
    
    # Initialize preprocessor
    preprocessor = FashionDataPreprocessor(RAW_DATA_PATH, PROCESSED_DATA_PATH)
    
    # Run pipeline
    stats = preprocessor.run_full_pipeline()
    
    print("\n🎉 All done! Your dataset is ready for training.")


if __name__ == "__main__":
    main()