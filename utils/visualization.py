#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization utilities for rail track defect detection model.
"""

import matplotlib.pyplot as plt


def show_train_history(train, test, label):
    """
    Plot training history.
    
    Args:
        train (list): Training metrics
        test (list): Testing metrics
        label (str): Label for the y-axis
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train)
    plt.plot(test)
    plt.title('Training History')
    plt.ylabel(label)
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.grid(True)
    plt.show()


def plot_learning_rate(lr_list):
    """
    Plot learning rate schedule.
    
    Args:
        lr_list (list): List of learning rates
        
    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lr_list)
    plt.xlabel('Iteration')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    plt.show()


def plot_sample_images(dataloader, num_images=5):
    """
    Plot sample images from a dataloader.
    
    Args:
        dataloader (DataLoader): Data loader
        num_images (int): Number of images to plot
        
    Returns:
        None
    """
    # Get a batch of images
    images, labels = next(iter(dataloader))
    
    # Denormalize images
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Create figure
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    
    # Plot images
    for i in range(num_images):
        # Denormalize image
        img = images[i].numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Plot image
        axes[i].imshow(img)
        axes[i].set_title(f"Label: {'Defective' if labels[i] == 0 else 'Non-Defective'}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix.
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        class_names (list): List of class names
        
    Returns:
        None
    """
    import numpy as np
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.show()
