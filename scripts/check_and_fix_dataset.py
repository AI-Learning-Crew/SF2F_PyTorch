#!/usr/bin/env python3
"""
Script to check and clean corrupted pickle files in the dataset
"""

import os
import pickle
import shutil
from pathlib import Path
import argparse


def check_pickle_file(filepath):
    """
    Check if a pickle file is corrupted.
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        bool: True if the file is valid, False if corrupted
    """
    try:
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
            # Check if LogMel_Features key exists
            if 'LogMel_Features' not in data:
                print(f"Missing LogMel_Features key in {filepath}")
                return False
            return True
    except (EOFError, pickle.UnpicklingError, OSError, IOError, KeyError) as e:
        print(f"Corrupted file: {filepath} - {e}")
        return False


def scan_dataset(data_dir, fix_mode=False):
    """
    Scan the dataset directory to find corrupted files.
    
    Args:
        data_dir: Path to the dataset directory
        fix_mode: If True, delete corrupted files
    """
    corrupted_files = []
    total_files = 0
    
    # Scan vox1 and vox2 directories
    for sub_dataset in ['vox1', 'vox2']:
        mel_dir = os.path.join(data_dir, sub_dataset, 'mel_spectrograms')
        
        if not os.path.exists(mel_dir):
            print(f"Directory not found: {mel_dir}")
            continue
            
        print(f"Scanning {mel_dir}...")
        
        for person_dir in os.listdir(mel_dir):
            person_path = os.path.join(mel_dir, person_dir)
            
            if not os.path.isdir(person_path):
                continue
                
            for pickle_file in os.listdir(person_path):
                if not pickle_file.endswith('.pickle'):
                    continue
                    
                pickle_path = os.path.join(person_path, pickle_file)
                total_files += 1
                
                if not check_pickle_file(pickle_path):
                    corrupted_files.append(pickle_path)
                    
                    if fix_mode:
                        try:
                            os.remove(pickle_path)
                            print(f"Deleted corrupted file: {pickle_path}")
                        except Exception as e:
                            print(f"Failed to delete {pickle_path}: {e}")
    
    print(f"\n=== Scan Results ===")
    print(f"Total files: {total_files}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Corruption rate: {len(corrupted_files)/total_files*100:.2f}%")
    
    if corrupted_files and not fix_mode:
        print(f"\nCorrupted files list (first 10):")
        for i, file in enumerate(corrupted_files[:10]):
            print(f"  {i+1}. {file}")
        if len(corrupted_files) > 10:
            print(f"  ... and {len(corrupted_files)-10} more")
            
        print(f"\nTo delete corrupted files, use --fix option:")
        print(f"python {__file__} --data_dir {data_dir} --fix")
    
    return corrupted_files


def main():
    parser = argparse.ArgumentParser(description='Check and clean corrupted pickle files in dataset')
    parser.add_argument('--data_dir', type=str, default='./data/VoxCeleb', 
                       help='Path to VoxCeleb dataset directory')
    parser.add_argument('--fix', action='store_true', 
                       help='Delete corrupted files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return
    
    print(f"Starting dataset scan: {args.data_dir}")
    if args.fix:
        print("WARNING: This will delete corrupted files!")
        response = input("Do you want to continue? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled.")
            return
    
    scan_dataset(args.data_dir, args.fix)
    
    if args.fix:
        print("\nCleanup completed! You can now retry training.")


if __name__ == '__main__':
    main() 