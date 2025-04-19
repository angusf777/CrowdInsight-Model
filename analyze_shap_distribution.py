#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analyze SHAP value distributions across the Kickstarter dataset.

This script generates and analyzes SHAP (SHapley Additive exPlanations) values
for a batch of Kickstarter projects to understand feature importance patterns.
It processes data in batches, calculates SHAP values for each sample, and produces
statistical analyses and visualizations of feature importance distributions.
Results help identify which features consistently impact model predictions.

Example usage:
    python analyze_shap_distribution.py --input_json data/allProcessed.json

Author: Angus Fung
Date: April 2025
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import argparse
from collections import defaultdict

from src.model import KickstarterModel
from src.explainer import KickstarterExplainer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze SHAP value distribution for all features")
    
    # Input parameters
    parser.add_argument("--input_json", type=str, default="data/allProcessed.json", 
                       help="Input JSON file path (default: data/allProcessed.json)")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="models/hparam_run_242/best_model.pth", 
                       help="Model checkpoint path (default: models/hparam_run_242/best_model.pth)")
    parser.add_argument("--hidden_dim", type=int, default=256, 
                       help="Hidden dimension size (default: 256)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="analysis/shap_distribution", 
                       help="Output directory (default: analysis/shap_distribution)")
    
    # Batch size for processing
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for processing (default: 32)")
    
    # Maximum number of samples to process
    parser.add_argument("--max_samples", type=int, default=None, 
                       help="Maximum number of samples to process (default: all)")
    
    return parser.parse_args()

def process_data_in_batches(data, batch_size):
    """Process data in batches to avoid memory issues"""
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

def prepare_input_batch(batch_data):
    """Prepare input batch for model"""
    # Initialize batch inputs
    batch_inputs = {
        'description_embedding': [],
        'blurb_embedding': [],
        'risk_embedding': [],
        'subcategory_embedding': [],
        'category_embedding': [],
        'country_embedding': [],
        'numerical_features': []
    }
    
    # Process each sample in the batch
    for sample in batch_data:
        # Process embedding vectors
        for embedding_name in ['description_embedding', 'blurb_embedding', 'risk_embedding', 
                              'subcategory_embedding', 'category_embedding', 'country_embedding']:
            if embedding_name in sample:
                batch_inputs[embedding_name].append(sample[embedding_name])
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
                batch_inputs[embedding_name].append([0.0] * dim)
        
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
        batch_inputs['numerical_features'].append(numerical_features)
    
    # Convert to tensors
    batch_tensors = {}
    for key, value in batch_inputs.items():
        batch_tensors[key] = torch.tensor(value, dtype=torch.float32)
    
    return batch_tensors

def main():
    """Main function"""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading data from: {args.input_json}")
    try:
        # Load input data
        with open(args.input_json, 'r') as f:
            input_data = json.load(f)
        print(f"Loaded {len(input_data)} samples")
        
        # Limit samples if requested
        if args.max_samples is not None:
            input_data = input_data[:args.max_samples]
            print(f"Limited to {len(input_data)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    print(f"Loading model from: {args.model_path}")
    try:
        # Load model with specified hidden dimension
        model = KickstarterModel(hidden_dim=args.hidden_dim)
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize explainer
    explainer = KickstarterExplainer(model, device)
    
    # Store SHAP values for all features
    all_shap_values = defaultdict(list)
    all_predictions = []
    
    print("Processing samples...")
    
    # Process data in batches
    batch_count = 0
    for batch_data in tqdm(process_data_in_batches(input_data, args.batch_size), 
                          total=(len(input_data) + args.batch_size - 1) // args.batch_size):
        batch_count += 1
        
        # Prepare input batch
        batch_inputs = prepare_input_batch(batch_data)
        
        # Move to device
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        
        # Process each sample in the batch
        for i in range(len(batch_data)):
            # Extract single sample
            sample_inputs = {k: v[i:i+1] for k, v in batch_inputs.items()}
            
            # Get prediction and SHAP values
            try:
                prediction, shap_values = explainer.explain_prediction(sample_inputs)
                
                # Store prediction
                all_predictions.append(prediction)
                
                # Store SHAP values
                for feature, value in shap_values.items():
                    all_shap_values[feature].append(value)
            except Exception as e:
                print(f"Error processing sample {i} in batch {batch_count}: {e}")
                continue
        
        # Print progress
        if batch_count % 10 == 0:
            print(f"Processed {batch_count * args.batch_size} samples...")
    
    print(f"Processed {len(all_predictions)} samples successfully")
    
    # Convert to DataFrame for easier analysis
    shap_df = pd.DataFrame(all_shap_values)
    shap_df['prediction'] = all_predictions
    
    # Save raw data
    print("Saving raw SHAP data...")
    shap_df.to_csv(os.path.join(args.output_dir, "shap_values.csv"), index=False)
    
    # Basic statistics
    print("\nFeature importance statistics:")
    stats = shap_df.describe()
    print(stats)
    
    # Save statistics
    stats.to_csv(os.path.join(args.output_dir, "shap_statistics.csv"))
    
    # Plot SHAP value distributions for each feature
    print("\nCreating SHAP distribution plots...")
    
    # Overall feature importance (mean absolute SHAP value)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    mean_abs_shap = mean_abs_shap[mean_abs_shap.index != 'prediction']
    
    plt.figure(figsize=(12, 8))
    mean_abs_shap.plot(kind='bar')
    plt.title('Mean Absolute SHAP Values (Feature Importance)')
    plt.ylabel('Mean |SHAP Value|')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "mean_absolute_shap.png"))
    
    # Distribution plots for each feature
    for feature in all_shap_values.keys():
        plt.figure(figsize=(10, 6))
        sns.histplot(all_shap_values[feature], kde=True)
        plt.title(f'SHAP Value Distribution: {feature}')
        plt.xlabel('SHAP Value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"shap_dist_{feature}.png"))
    
    # Combined SHAP distribution plot
    plt.figure(figsize=(14, 10))
    for feature in list(all_shap_values.keys())[:10]:  # Limit to top 10 features for clarity
        sns.kdeplot(all_shap_values[feature], label=feature)
    plt.title('SHAP Value Distributions (Top 10 Features)')
    plt.xlabel('SHAP Value')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "combined_shap_dist.png"))
    
    # SHAP values heatmap (correlation)
    plt.figure(figsize=(12, 10))
    feature_corr = shap_df.corr()
    sns.heatmap(feature_corr, annot=False, cmap='coolwarm')
    plt.title('SHAP Value Correlation Between Features')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "shap_correlation.png"))
    
    # SHAP values vs prediction
    plt.figure(figsize=(12, 8))
    for feature in list(all_shap_values.keys())[:5]:  # Limit to top 5 features for clarity
        plt.scatter(all_predictions, all_shap_values[feature], alpha=0.2, label=feature)
    plt.title('SHAP Values vs Prediction Probability')
    plt.xlabel('Prediction Probability')
    plt.ylabel('SHAP Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "shap_vs_prediction.png"))
    
    # Violin plots for feature distributions
    plt.figure(figsize=(15, 10))
    shap_melted = pd.melt(shap_df.drop(columns=['prediction']), var_name='Feature', value_name='SHAP Value')
    sns.violinplot(x='Feature', y='SHAP Value', data=shap_melted)
    plt.xticks(rotation=45, ha='right')
    plt.title('SHAP Value Distributions Across Features')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "shap_violinplot.png"))
    
    print(f"\nAll analysis results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 