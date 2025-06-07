#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training utilities for rail track defect detection model.
"""

import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR


def train_model(model, train_loader, test_loader, device, num_epochs, max_lr=3e-3):
    """
    Train the model with frozen backbone.
    
    Args:
        model (torch.nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Testing data loader
        device (torch.device): Device to train on
        num_epochs (int): Number of training epochs
        max_lr (float): Maximum learning rate
        
    Returns:
        tuple: (model, history_dict)
    """
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Set up the 1cycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=num_epochs)
    
    # Initialize tracking variables
    lr_list = []
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    epoch_times = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())  # Convert labels to float
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track training loss
            running_loss += loss.item() * inputs.size(0)
            
            # Track accuracy
            predicted = (outputs >= 0.5).float()
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Append the current learning rate
            current_lr = scheduler.get_last_lr()[0]
            lr_list.append(current_lr)
        
        # Calculate epoch metrics
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_predictions / total_predictions
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        test_running_loss = 0.0
        test_correct_predictions = 0
        test_total_predictions = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
                
                # Track validation loss
                test_running_loss += loss.item() * inputs.size(0)
                
                # Track validation accuracy
                predicted = (outputs >= 0.5).float()
                test_total_predictions += labels.size(0)
                test_correct_predictions += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        test_loss = test_running_loss / len(test_loader.dataset)
        test_accuracy = test_correct_predictions / test_total_predictions
        
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Calculate epoch time
        end_time = time.time()
        epoch_time = end_time - start_time
        epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, "
              f"Test Accuracy: {test_accuracy:.4f}, Time: {epoch_time:.2f}s")
    
    # Calculate average epoch time
    avg_epoch_time = torch.mean(torch.tensor(epoch_times))
    std_epoch_time = torch.std(torch.tensor(epoch_times))
    print(f"Average epoch time: {avg_epoch_time:.2f}s ± {std_epoch_time:.2f}s")
    
    # Return history
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'lr_list': lr_list,
        'epoch_times': epoch_times,
        'avg_epoch_time': avg_epoch_time.item(),
        'std_epoch_time': std_epoch_time.item()
    }
    
    return model, history


def fine_tune_model(model, train_loader, test_loader, device, num_epochs, max_lr=1e-3):
    """
    Fine-tune the model with unfrozen backbone.
    
    Args:
        model (torch.nn.Module): The model to fine-tune
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Testing data loader
        device (torch.device): Device to train on
        num_epochs (int): Number of fine-tuning epochs
        max_lr (float): Maximum learning rate
        
    Returns:
        tuple: (model, history_dict)
    """
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Set up the 1cycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=num_epochs)
    
    # Initialize tracking variables
    lr_list = []
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Fine-tuning Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track training loss
            running_loss += loss.item() * inputs.size(0)
            
            # Track accuracy
            predicted = (outputs >= 0.5).float()
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Append the current learning rate
            current_lr = scheduler.get_last_lr()[0]
            lr_list.append(current_lr)
        
        # Calculate epoch metrics
        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = correct_predictions / total_predictions
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        test_running_loss = 0.0
        test_correct_predictions = 0
        test_total_predictions = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels.float())
                
                # Track validation loss
                test_running_loss += loss.item() * inputs.size(0)
                
                # Track validation accuracy
                predicted = (outputs >= 0.5).float()
                test_total_predictions += labels.size(0)
                test_correct_predictions += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        test_loss = test_running_loss / len(test_loader.dataset)
        test_accuracy = test_correct_predictions / test_total_predictions
        
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print(f"Fine-tuning Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, "
              f"Test Accuracy: {test_accuracy:.4f}")
    
    # Return history
    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'lr_list': lr_list
    }
    
    return model, history
