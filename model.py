#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DaViT model implementation for rail track defect detection.
"""

import torch
import torch.nn as nn
import timm


def create_railtrack_davit_model(model_name, device):
    """
    Create and initialize a RailTrack-DaViT model for binary classification.
    
    Args:
        model_name (str): Name of the DaViT model variant from timm
        device (torch.device): Device to move the model to
        
    Returns:
        torch.nn.Module: The initialized model
    """
    # Load the pre-trained DaViT model
    base_model = timm.create_model(model_name, pretrained=True)
    
    # Freeze the parameters of the base model
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Get the number of features from the head
    num_features = base_model.head.in_features
    print(f"Input features: {num_features}")
    
    # Classifier head
    base_model.head.fc = nn.Sequential(
        nn.Linear(num_features, 512),  # Features from DaViT
        nn.ELU(),
        nn.Dropout(0.5),
        nn.Linear(512, 128),
        nn.ELU(),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )
    
    # Unfreeze classifier
    for param in base_model.head.fc.parameters():
        param.requires_grad = True
    
    # Move model to device
    model = base_model.to(device)
    
    return model


def unfreeze_model(model):
    """
    Unfreeze all parameters in the model for fine-tuning.
    
    Args:
        model (torch.nn.Module): The model to unfreeze
        
    Returns:
        torch.nn.Module: The unfrozen model
    """
    for param in model.parameters():
        param.requires_grad = True
    
    return model
