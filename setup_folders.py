"""
setup_folders.py
================
Temporal Fashion AI System - Folder Structure Generator

Purpose: 
    Creates the complete folder structure for the project. 
    Ensures all necessary directories exist for development.

Usage:
    python setup_folders.py

Author:  Pabasara11
Date: 2025-12-18
"""

import os
from pathlib import Path

def create_folder_structure():
    """Create all necessary folders for the project."""
    
    # Define folder structure
    folders = [
        # Backend Structure
        "backend/models",
        "backend/api",
        "backend/core",
        "backend/utils",
        "backend/data/logs",
        
        # Frontend Structure
        "frontend/src/components",
        "frontend/src/pages",
        "frontend/src/api",
        "frontend/src/styles",
        "frontend/public",
        
        # Notebooks for Training
        "notebooks",
        
        # Data Folders
        "data/raw/fashion-dataset",
        "data/raw/accessories",
        "data/processed/outfits",
        "data/processed/accessories",
        "data/processed/annotations",
        "data/synthetic/user_histories",
        "data/synthetic/preference_trajectories",
        
        # Documentation
        "docs/api",
        "docs/models",
        "docs/architecture",
        
        # Tests
        "tests/unit",
        "tests/integration",
    ]
    
    print("=" * 60)
    print("🚀 TEMPORAL FASHION AI - Folder Structure Setup")
    print("=" * 60)
    print()
    
    created_count = 0
    existing_count = 0
    
    for folder in folders:
        folder_path = Path(folder)
        
        if folder_path.exists():
            print(f"✓ Already exists: {folder}")
            existing_count += 1
        else:
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {folder}")
            created_count += 1
    
    print()
    print("=" * 60)
    print(f"📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Folders created: {created_count}")
    print(f"✓  Already existed: {existing_count}")
    print(f"📁 Total folders: {len(folders)}")
    print()
    print("🎉 Folder structure setup complete!")
    print()
    
    # Display folder tree
    print("=" * 60)
    print("📂 PROJECT STRUCTURE")
    print("=" * 60)
    print()
    print("temporal-fashion-ai/")
    for folder in sorted(folders):
        depth = folder.count('/')
        indent = "│   " * depth
        folder_name = folder.split('/')[-1]
        print(f"{indent}├── {folder_name}/")
    print()

if __name__ == "__main__":
    create_folder_structure()