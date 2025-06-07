#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Dataset handling for rail track defect detection.
"""

import os
import torch
from torchvision import datasets, transforms


def create_transforms(image_size):
    """
    Create data transformations for training and testing.
    
    Args:
        image_size (int): Target image size for the model
        
    Returns:
        tuple: (train_transforms, test_transforms)
    """
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(40),
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, test_transforms


def create_dataloaders(train_dir, test_dir, image_size, batch_size):
    """
    Create data loaders for training and testing.
    
    Args:
        train_dir (str): Directory containing training data
        test_dir (str): Directory containing testing data
        image_size (int): Target image size for the model
        batch_size (int): Batch size for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Create transforms
    train_transforms, test_transforms = create_transforms(image_size)
    
    # Create datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_data, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    print(f"Training samples: {len(train_data)}")
    print(f"Testing samples: {len(test_data)}")
    
    # Print class distribution
    train_defective = len(os.listdir(os.path.join(train_dir, 'Defective')))
    train_nondefective = len(os.listdir(os.path.join(train_dir, 'Non Defective')))
    test_defective = len(os.listdir(os.path.join(test_dir, 'Defective')))
    test_nondefective = len(os.listdir(os.path.join(test_dir, 'Non Defective')))
    
    print(f"Training - Defective: {train_defective}, Non-Defective: {train_nondefective}")
    print(f"Testing - Defective: {test_defective}, Non-Defective: {test_nondefective}")
    
    return train_loader, test_loader


def download_dataset(file_id, output_file):
    """
    Download dataset from Google Drive.
    
    Args:
        file_id (str): Google Drive file ID
        output_file (str): Output file name
        
    Returns:
        None
    """
    import gdown
    
    # Download the dataset
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_file)
    
    # Extract the dataset
    import zipfile
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    print(f"Dataset downloaded and extracted from {output_file}")
