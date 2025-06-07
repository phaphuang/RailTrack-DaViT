#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation utilities for rail track defect detection model.
"""

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test set.
    
    Args:
        model (torch.nn.Module): The trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to evaluate on
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Initialize lists to store true labels and predicted labels
    test_labels = []
    predicted_labels = []
    
    # Disable gradient computation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()  # Remove extra dimension
            predicted = (outputs >= 0.5).float()
            test_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    
    # Convert lists to numpy arrays
    test_labels = np.array(test_labels)
    predicted_labels = np.array(predicted_labels)
    
    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(test_labels, predicted_labels).ravel()
    print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
    
    # Define class labels
    labels = ['Defective', 'Non Defective']
    
    # Calculate confusion matrix
    cm = confusion_matrix(test_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate classification report
    report = classification_report(test_labels, predicted_labels, target_names=labels)
    print("Classification Report:")
    print(report)
    
    # Calculate additional metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return metrics
