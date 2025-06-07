#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Saving utilities for rail track defect detection model.
"""

import os
import torch
import pandas as pd


def save_model(model, model_name, dataset_name, image_size, num_epochs, fine_tune_epochs, output_dir):
    """
    Save the trained model.
    
    Args:
        model (torch.nn.Module): The trained model
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        image_size (int): Image size
        num_epochs (int): Number of training epochs
        fine_tune_epochs (int): Number of fine-tuning epochs
        output_dir (str): Directory to save the model
        
    Returns:
        str: Path to the saved model
    """
    # Create model directory
    model_name_base = model_name.split(".")[0]
    root_path = os.path.join(output_dir, model_name_base)
    model_root_path = os.path.join(
        root_path, 
        f"railtrack-{dataset_name}-{model_name_base}-pytorch-sz{image_size}-"
        f"ep{num_epochs}-random-unfreeze{fine_tune_epochs}ep-modfc-dropout-elu-onecyclelr"
    )
    
    # Create directories
    os.makedirs(root_path, exist_ok=True)
    os.makedirs(model_root_path, exist_ok=True)
    
    # Save model
    model_filename = f"{model_name_base}_sz{image_size}_ep{num_epochs}.pth"
    model_path = os.path.join(model_root_path, model_filename)
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved as {model_path}")
    
    return model_root_path


def save_results(train_history, fine_tune_history, model_name, dataset_name, 
                image_size, num_epochs, fine_tune_epochs, model_path):
    """
    Save training results.
    
    Args:
        train_history (dict): Training history
        fine_tune_history (dict): Fine-tuning history
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        image_size (int): Image size
        num_epochs (int): Number of training epochs
        fine_tune_epochs (int): Number of fine-tuning epochs
        model_path (str): Path to the saved model
        
    Returns:
        None
    """
    # Save epoch times
    with open(os.path.join(model_path, 'epochtime_training_log.txt'), 'w') as f:
        f.write(f"Epoch times: {train_history['epoch_times']}\n")
        f.write(f"Average training time per epoch: {train_history['avg_epoch_time']:.3f} seconds\n")
        f.write(f"Standard deviation of epoch times: {train_history['std_epoch_time']:.3f} seconds\n")
    
    # Save training results
    model_name_base = model_name.split(".")[0]
    
    # Initial training results
    epoch_list = list(range(1, num_epochs + 1))
    data = []
    for ep, tl, ta, tel, tea in zip(
        epoch_list, 
        train_history['train_losses'], 
        train_history['train_accuracies'], 
        train_history['test_losses'], 
        train_history['test_accuracies']
    ):
        data.append([ep, tl, ta, tel, tea])
    
    # Create DataFrame
    df = pd.DataFrame(
        data, 
        columns=["Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"]
    )
    
    # Save to CSV
    csv_filename = f"{dataset_name}_accuracy_loss_{model_name_base}_sz{image_size}_ep{num_epochs}.csv"
    df.to_csv(os.path.join(model_path, csv_filename), index=False)
    print(f"Training data exported to {csv_filename}")
    
    # Fine-tuning results
    epoch_list = list(range(1, fine_tune_epochs + 1))
    data = []
    for ep, tl, ta, tel, tea in zip(
        epoch_list, 
        fine_tune_history['train_losses'], 
        fine_tune_history['train_accuracies'], 
        fine_tune_history['test_losses'], 
        fine_tune_history['test_accuracies']
    ):
        data.append([ep, tl, ta, tel, tea])
    
    # Create DataFrame
    df2 = pd.DataFrame(
        data, 
        columns=["Epoch", "Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"]
    )
    
    # Save to CSV
    csv_filename = f"{dataset_name}_unfreeze{fine_tune_epochs}ep_accuracy_loss_{model_name_base}_sz{image_size}_ep{fine_tune_epochs}.csv"
    df2.to_csv(os.path.join(model_path, csv_filename), index=False)
    print(f"Fine-tuning data exported to {csv_filename}")
