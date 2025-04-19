#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate the trained Kickstarter Project Success Prediction Model.

This script performs comprehensive evaluation of the trained model's performance
on test data. It calculates metrics including ROC-AUC, Brier Score, Log Loss,
accuracy, F1 score, precision, and recall. The script also generates visualizations
of model performance and conducts SHAP analysis to understand feature importance.
Results are saved to the evaluation directory.

Example usage:
    python testing_analysis.py

Author: Angus Fung
Date: April 2025
"""

import torch
import json
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, precision_recall_fscore_support, accuracy_score, roc_curve, confusion_matrix

from src.model import KickstarterModel
from src.data_processor import KickstarterDataProcessor

def load_model():
    """Load model"""
    print("Loading best model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model with hidden_dim=256 to match the saved model
    model = KickstarterModel(hidden_dim=256)
    
    # Load checkpoint and get model state
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    
    # Check if in checkpoint format
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'N/A')
        val_auc = checkpoint.get('val_auc', 'N/A')
        val_brier = checkpoint.get('val_brier', 'N/A')
        
        print(f"Model from Epoch: {epoch}")
        print(f"Validation AUC: {val_auc}")
        print(f"Validation Brier Score: {val_brier}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, device

def load_test_data(test_file='data/test_data.json'):
    """Load test data from JSON file"""
    print(f"Loading test data from {test_file}...")
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data file not found: {test_file}")
    
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test samples")
    return test_data

def preprocess_test_data(test_data):
    """Process test data into PyTorch tensors"""
    processed_data = []
    
    for sample in test_data:
        # Process embedding vectors
        processed_sample = {}
        
        for embedding_name in ['description_embedding', 'blurb_embedding', 'risk_embedding', 
                              'subcategory_embedding', 'category_embedding', 'country_embedding']:
            if embedding_name in sample:
                processed_sample[embedding_name] = torch.tensor([sample[embedding_name]], dtype=torch.float32)
            else:
                # Use appropriate zero vector
                if embedding_name == 'description_embedding':
                    dim = 768
                elif embedding_name in ['blurb_embedding', 'risk_embedding']:
                    dim = 384
                elif embedding_name == 'subcategory_embedding':
                    dim = 100
                elif embedding_name == 'category_embedding':
                    dim = 15
                else:  # country_embedding
                    dim = 100
                processed_sample[embedding_name] = torch.zeros((1, dim), dtype=torch.float32)
        
        # Process numerical features
        numerical_features = [
            sample.get('description_length', 0),
            sample.get('funding_goal', 0),
            sample.get('image_count', 0),
            sample.get('video_count', 0),
            sample.get('campaign_duration', 0),
            sample.get('previous_projects_count', 0),
            sample.get('previous_success_rate', 0),
            sample.get('previous_pledged', 0),
            sample.get('previous_funding_goal', 0)
        ]
        processed_sample['numerical_features'] = torch.tensor([numerical_features], dtype=torch.float32)
        
        # Process label
        if 'state' in sample:
            processed_sample['label'] = torch.tensor([float(sample['state'])], dtype=torch.float32)
        
        processed_data.append(processed_sample)
    
    return processed_data

def calculate_shap_values(model, inputs, device):
    """Calculate SHAP values for the model inputs"""
    # Define feature names
    feature_names = [
        'description_embedding', 'blurb_embedding', 'risk_embedding', 
        'subcategory_embedding', 'category_embedding', 'country_embedding',
        'description_length', 'funding_goal', 'image_count', 'video_count', 'campaign_duration', 
        'previous_projects_count', 'previous_success_rate', 'previous_pledged', 'previous_funding_goal'
    ]
    
    # Define embedding map
    embedding_map = {
        'description_embedding': 'description_embedding', 
        'blurb_embedding': 'blurb_embedding', 
        'risk_embedding': 'risk_embedding', 
        'subcategory_embedding': 'subcategory_embedding', 
        'category_embedding': 'category_embedding', 
        'country_embedding': 'country_embedding'
    }
    
    # Define numerical features
    numerical_features = [
        'description_length', 'funding_goal', 'image_count', 'video_count', 'campaign_duration', 
        'previous_projects_count', 'previous_success_rate', 'previous_pledged', 'previous_funding_goal'
    ]
    
    # Initialize SHAP values dictionary
    all_shap_values = {feature: [] for feature in feature_names}
    
    # Calculate baseline prediction
    baseline = {k: torch.zeros_like(v) for k, v in inputs.items() if k != 'label'}
    baseline_pred, _ = model(baseline)
    
    # Calculate SHAP values for each feature
    for feature_name, embedding_name in embedding_map.items():
        if embedding_name in inputs:
            feature_input = {k: torch.zeros_like(v) for k, v in inputs.items() if k != 'label'}
            feature_input[embedding_name] = inputs[embedding_name]
            feature_pred, _ = model(feature_input)
            shap = (feature_pred - baseline_pred).cpu().numpy()
            all_shap_values[feature_name].append(float(shap))
    
    # Process numerical features
    if 'numerical_features' in inputs:
        for idx, feature_name in enumerate(numerical_features):
            feature_input = {k: torch.zeros_like(v) for k, v in inputs.items() if k != 'label'}
            feature_input['numerical_features'] = torch.zeros_like(inputs['numerical_features'])
            feature_input['numerical_features'][:, idx] = inputs['numerical_features'][:, idx]
            feature_pred, _ = model(feature_input)
            shap = (feature_pred - baseline_pred).cpu().numpy()
            all_shap_values[feature_name].append(float(shap))
    
    return all_shap_values

def find_optimal_threshold(y_true, y_pred):
    """Find the optimal threshold to maximize f1 score"""
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_binary = (np.array(y_pred) >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average='binary')
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

def evaluate_model():
    """Evaluate the model on the test dataset and save results to evaluation/accuracy_results.json"""
    print("Starting comprehensive model evaluation...")
    
    # Load model
    model, device = load_model()
    
    # Load and process test data
    test_data = load_test_data()
    processed_data = preprocess_test_data(test_data)
    
    # Tracking variables
    all_probs = []
    all_labels = []
    all_shap_values = {}
    
    # Initialize SHAP values
    feature_names = [
        'description_embedding', 'blurb_embedding', 'risk_embedding', 
        'subcategory_embedding', 'category_embedding', 'country_embedding',
        'description_length', 'funding_goal', 'image_count', 'video_count', 'campaign_duration', 
        'previous_projects_count', 'previous_success_rate', 'previous_pledged', 'previous_funding_goal'
    ]
    for feature in feature_names:
        all_shap_values[feature] = []
    
    # Evaluate each sample
    with torch.no_grad():
        for sample in tqdm(processed_data, desc="Evaluating test samples"):
            # Move data to device
            inputs = {k: v.to(device) for k, v in sample.items() if k != 'label'}
            
            # Make prediction
            probs, _ = model(inputs)
            all_probs.append(float(probs.cpu().numpy()))
            
            # Get label if available
            if 'label' in sample:
                label = sample['label'].item()
                all_labels.append(label)
            
            # Calculate SHAP values
            sample_shap_values = calculate_shap_values(model, inputs, device)
            for feature in feature_names:
                if feature in sample_shap_values and sample_shap_values[feature]:
                    all_shap_values[feature].extend(sample_shap_values[feature])
    
    # Calculate optimal threshold
    if all_labels:
        optimal_threshold = find_optimal_threshold(all_labels, all_probs)
        print(f"Optimal threshold: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.5
        print("No labels available, using default threshold of 0.5")
    
    # Calculate metrics
    results = {}
    results["threshold"] = float(optimal_threshold)
    
    if all_labels:
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred_proba = np.array(all_probs)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred) * 100
        results["accuracy"] = float(accuracy)
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results["confusion_matrix"] = {
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        results["roc_auc"] = float(roc_auc)
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # Brier Score
        brier = brier_score_loss(y_true, y_pred_proba)
        results["brier_score"] = float(brier)
        print(f"Brier Score: {brier:.4f}")
        
        # Log Loss
        ll = log_loss(y_true, y_pred_proba)
        results["log_loss"] = float(ll)
        print(f"Log Loss: {ll:.4f}")
        
        # Success rate
        success_rate = np.mean(y_true) * 100
        results["success_rate"] = {
            "actual": float(success_rate),
            "predicted": float(np.mean(y_pred) * 100)
        }
        print(f"Actual Success Rate: {success_rate:.2f}%")
        print(f"Predicted Success Rate: {np.mean(y_pred) * 100:.2f}%")
        
        # Generate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Ensure evaluation directory exists
        os.makedirs('evaluation', exist_ok=True)
        plt.savefig('evaluation/roc_curve.png')
        plt.close()
    
    # Calculate SHAP statistics
    shap_stats = {}
    for feature in feature_names:
        if feature in all_shap_values and all_shap_values[feature]:
            values = all_shap_values[feature]
            shap_stats[feature] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "positive_ratio": float(np.mean([1 if v > 0 else 0 for v in values])),
                "absolute_mean": float(np.mean(np.abs(values)))
            }
    
    # Sort features by importance (absolute mean)
    sorted_features = sorted(shap_stats.items(), key=lambda x: x[1]['absolute_mean'], reverse=True)
    sorted_features_list = [f for f, _ in sorted_features]
    
    results["shap_stats"] = shap_stats
    results["sorted_features_by_importance"] = sorted_features_list
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    feature_importance = {f: shap_stats[f]['absolute_mean'] for f in sorted_features_list[:10]}  # Top 10 features
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    colors = ['green' if shap_stats[f]['mean'] > 0 else 'red' for f in features]
    
    plt.barh(range(len(features)), importance, color=colors, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('evaluation/feature_importance.png')
    plt.close()
    
    # Save results to JSON
    with open('evaluation/accuracy_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to evaluation/accuracy_results.json")
    print(f"ROC curve saved to evaluation/roc_curve.png")
    print(f"Feature importance plot saved to evaluation/feature_importance.png")
    
    return results

if __name__ == "__main__":
    evaluate_model() 