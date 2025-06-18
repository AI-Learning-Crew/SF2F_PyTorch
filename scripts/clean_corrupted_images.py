#!/usr/bin/env python3
"""
Clean corrupted images from VoxCeleb dataset
Removes 0-byte files and images that cannot be opened by PIL
"""

import os
import argparse
from PIL import Image
import shutil
from pathlib import Path

def check_image_validity(image_path):
    """
    Check if an image file is valid
    Returns (is_valid, error_message)
    """
    try:
        # Check file size first
        if os.path.getsize(image_path) == 0:
            return False, "0-byte file"
        
        # Try to open with PIL
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                img.verify()  # Verify image integrity
                return True, None
    except Exception as e:
        return False, str(e)

def find_corrupted_images(dataset_dir, face_type='masked'):
    """
    Find all corrupted images in the dataset
    Returns list of (file_path, error_reason)
    """
    corrupted_files = []
    face_dir = f"{face_type}_faces"
    
    print(f"Scanning for corrupted images in {dataset_dir}...")
    
    for sub_dataset in ['vox1', 'vox2']:
        faces_path = os.path.join(dataset_dir, sub_dataset, face_dir)
        
        if not os.path.exists(faces_path):
            print(f"WARNING: {faces_path} does not exist, skipping...")
            continue
            
        print(f"Checking {faces_path}...")
        
        # Walk through all subdirectories
        for person_name in os.listdir(faces_path):
            person_dir = os.path.join(faces_path, person_name)
            
            if not os.path.isdir(person_dir):
                continue
                
            # Check each image file
            for image_file in os.listdir(person_dir):
                if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                    
                image_path = os.path.join(person_dir, image_file)
                is_valid, error_msg = check_image_validity(image_path)
                
                if not is_valid:
                    corrupted_files.append((image_path, error_msg))
                    
        print(f"Completed checking {sub_dataset}")
    
    return corrupted_files

def remove_corrupted_images(corrupted_files, backup_dir=None, dry_run=False):
    """
    Remove corrupted image files
    """
    if not corrupted_files:
        print("No corrupted files found!")
        return
    
    print(f"\nFound {len(corrupted_files)} corrupted files:")
    
    # Create backup directory if specified
    if backup_dir and not dry_run:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"Backup directory created: {backup_dir}")
    
    removed_count = 0
    backup_count = 0
    
    for file_path, error_msg in corrupted_files:
        print(f"  {file_path} - {error_msg}")
        
        if dry_run:
            print(f"    [DRY RUN] Would remove: {file_path}")
            continue
            
        try:
            # Backup file if backup directory is specified
            if backup_dir:
                # Create relative path structure in backup
                rel_path = os.path.relpath(file_path, start=os.path.commonpath([file_path]))
                backup_path = os.path.join(backup_dir, rel_path)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.move(file_path, backup_path)
                backup_count += 1
                print(f"    Moved to backup: {backup_path}")
            else:
                # Direct removal
                os.remove(file_path)
                removed_count += 1
                print(f"    Removed: {file_path}")
                
        except Exception as e:
            print(f"    ERROR: Failed to process {file_path}: {e}")
    
    if not dry_run:
        if backup_dir:
            print(f"\nTotal files moved to backup: {backup_count}")
        else:
            print(f"\nTotal files removed: {removed_count}")
    else:
        print(f"\n[DRY RUN] Would process {len(corrupted_files)} files")

def clean_empty_directories(dataset_dir, face_type='masked', dry_run=False):
    """
    Remove empty person directories after cleaning
    """
    face_dir = f"{face_type}_faces"
    removed_dirs = []
    
    for sub_dataset in ['vox1', 'vox2']:
        faces_path = os.path.join(dataset_dir, sub_dataset, face_dir)
        
        if not os.path.exists(faces_path):
            continue
            
        for person_name in os.listdir(faces_path):
            person_dir = os.path.join(faces_path, person_name)
            
            if not os.path.isdir(person_dir):
                continue
                
            # Check if directory is empty
            if len(os.listdir(person_dir)) == 0:
                if dry_run:
                    print(f"[DRY RUN] Would remove empty directory: {person_dir}")
                else:
                    try:
                        os.rmdir(person_dir)
                        removed_dirs.append(person_dir)
                        print(f"Removed empty directory: {person_dir}")
                    except Exception as e:
                        print(f"ERROR: Failed to remove {person_dir}: {e}")
    
    if not dry_run and removed_dirs:
        print(f"\nRemoved {len(removed_dirs)} empty directories")

def main():
    parser = argparse.ArgumentParser(description='Clean corrupted images from VoxCeleb dataset')
    parser.add_argument('--dataset_dir', default='./data/VoxCeleb', 
                       help='Path to VoxCeleb dataset directory')
    parser.add_argument('--face_type', default='masked', choices=['masked', 'origin'],
                       help='Type of face images to clean')
    parser.add_argument('--backup_dir', 
                       help='Directory to backup corrupted files (if not specified, files will be deleted)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be done without actually removing files')
    parser.add_argument('--clean_empty_dirs', action='store_true',
                       help='Also remove empty person directories after cleaning')
    
    args = parser.parse_args()
    
    print("=== VoxCeleb Image Cleaner ===")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Face type: {args.face_type}")
    print(f"Backup directory: {args.backup_dir or 'None (files will be deleted)'}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Find corrupted images
    corrupted_files = find_corrupted_images(args.dataset_dir, args.face_type)
    
    # Remove or backup corrupted images
    remove_corrupted_images(corrupted_files, args.backup_dir, args.dry_run)
    
    # Clean empty directories if requested
    if args.clean_empty_dirs:
        print("\nCleaning empty directories...")
        clean_empty_directories(args.dataset_dir, args.face_type, args.dry_run)
    
    print("\nDone!")

if __name__ == '__main__':
    main() 