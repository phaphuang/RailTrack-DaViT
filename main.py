#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RailTrack-DaViT: Rail Track Defect Detection using DaViT Vision Transformer

This is the main entry point for training and evaluating the DaViT model
for rail track defect detection.
"""

import os
import argparse
import torch

from model import create_railtrack_davit_model
from dataset import create_dataloaders
from trainer import train_model, fine_tune_model
from evaluation import evaluate_model
from visualization import plot_learning_rate, show_train_history
from saving import save_model, save_results


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RailTrack-DaViT: Rail Track Defect Detection')
    
    # Dataset parameters
    parser.add_argument('--train_dir', type=str, default='train', help='Training directory')
    parser.add_argument('--test_dir', type=str, default='test', help='Testing directory')
    parser.add_argument('--image_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='davit_base.msft_in1k', 
                        help='Model name from timm')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=90, help='Number of training epochs')
    parser.add_argument('--fine_tune_epochs', type=int, default=10, 
                        help='Number of fine-tuning epochs')
    parser.add_argument('--lr', type=float, default=3e-3, help='Maximum learning rate')
    parser.add_argument('--fine_tune_lr', type=float, default=1e-3, 
                        help='Maximum fine-tuning learning rate')
    
    # Saving parameters
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Directory to save outputs')
    parser.add_argument('--dataset_name', type=str, default='3datasets', 
                        help='Name of the dataset')
    
    return parser.parse_args()


def main():
    """Main function to run the training and evaluation pipeline."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        args.train_dir, 
        args.test_dir, 
        args.image_size, 
        args.batch_size
    )
    
    # Create model
    model = create_railtrack_davit_model(args.model_name, device)
    
    # Train the model
    model, train_history = train_model(
        model, 
        train_loader, 
        test_loader, 
        device, 
        args.num_epochs, 
        args.lr
    )
    
    # Plot training history
    show_train_history(
        train_history['train_losses'], 
        train_history['test_losses'], 
        'loss'
    )
    show_train_history(
        train_history['train_accuracies'], 
        train_history['test_accuracies'], 
        'accuracy'
    )
    
    # Plot learning rate
    plot_learning_rate(train_history['lr_list'])
    
    # Fine-tune the model
    model, fine_tune_history = fine_tune_model(
        model, 
        train_loader, 
        test_loader, 
        device, 
        args.fine_tune_epochs, 
        args.fine_tune_lr
    )
    
    # Plot fine-tuning history
    show_train_history(
        fine_tune_history['train_losses'], 
        fine_tune_history['test_losses'], 
        'loss'
    )
    show_train_history(
        fine_tune_history['train_accuracies'], 
        fine_tune_history['test_accuracies'], 
        'accuracy'
    )
    
    # Plot fine-tuning learning rate
    plot_learning_rate(fine_tune_history['lr_list'])
    
    # Evaluate the model
    evaluate_model(model, test_loader, device)
    
    # Save the model and results
    model_path = save_model(
        model, 
        args.model_name, 
        args.dataset_name, 
        args.image_size, 
        args.num_epochs, 
        args.fine_tune_epochs, 
        args.output_dir
    )
    
    # Save training results
    save_results(
        train_history, 
        fine_tune_history, 
        args.model_name, 
        args.dataset_name, 
        args.image_size, 
        args.num_epochs, 
        args.fine_tune_epochs, 
        model_path
    )
    
    print(f"Training complete. Model saved at {model_path}")


if __name__ == "__main__":
    main()
