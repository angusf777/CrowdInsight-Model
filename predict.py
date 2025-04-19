#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate predictions using the trained Kickstarter Project Success Prediction Model.

This script loads a trained model and makes predictions on new Kickstarter project data.
It processes input data from a JSON file, generates success probability predictions,
and provides SHAP value explanations to interpret the model's decisions. Results are
saved as both JSON data and visualizations.

Example usage:
    python predict.py --input_json sample_input.json

Author: Angus Fung
Date: April 2025
"""

import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import KickstarterModel
from src.explainer import KickstarterExplainer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Kickstarter Project Success Prediction")
    
    # Input parameters
    parser.add_argument("--input_json", type=str, required=True, help="Input JSON file path")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, default="models/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size (default: 256)")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="predictions", help="Output directory")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load input data
    with open(args.input_json, 'r') as f:
        input_data = json.load(f)
    
    # Load model with specified hidden dimension
    model = KickstarterModel(hidden_dim=args.hidden_dim)
    print(f"Model initialized with hidden_dim={args.hidden_dim}")
    
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully: {args.model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Initialize explainer
    explainer = KickstarterExplainer(model, device)
    
    # Process input
    processed_inputs = {}
    
    # Process embedding vectors
    for embedding_name in ['description_embedding', 'blurb_embedding', 'risk_embedding', 
                           'subcategory_embedding', 'category_embedding', 'country_embedding']:
        if embedding_name in input_data:
            processed_inputs[embedding_name] = torch.tensor(input_data[embedding_name], dtype=torch.float32).unsqueeze(0)
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
            processed_inputs[embedding_name] = torch.zeros((1, dim), dtype=torch.float32)
    
    # Process numerical features
    numerical_features = [
        input_data.get('description_length', 0),
        input_data.get('funding_goal', 0),
        input_data.get('image_count', 0),
        input_data.get('video_count', 0),
        input_data.get('campaign_duration', 0),
        input_data.get('previous_projects_count', 0),
        input_data.get('previous_success_rate', 0),
        input_data.get('previous_pledged', 0),
        input_data.get('previous_funding_goal', 0)
    ]
    processed_inputs['numerical_features'] = torch.tensor([numerical_features], dtype=torch.float32)
    
    # Predict and explain
    prediction, shap_values = explainer.explain_prediction(processed_inputs)
    
    print(f"\nPrediction results:")
    print(f"Success probability: {prediction:.4f}")
    print(f"Prediction outcome: {'Success' if prediction >= 0.5 else 'Failure'}")
    
    print(f"\nFeature importance (SHAP values):")
    sorted_shap = dict(sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True))
    for feature, value in sorted_shap.items():
        print(f"{feature}: {value:.4f}")
    
    # Visualization
    output_path = os.path.join(args.output_dir, "prediction_shap.png")
    explainer.visualize_shap_values(shap_values, output_path)
    
    # Save results
    output_json_path = os.path.join(args.output_dir, "prediction_result.json")
    result = {
        "prediction": {
            "success_probability": float(prediction),
            "predicted_outcome": "Success" if prediction >= 0.5 else "Failure"
        },
        "explanation": {
            "shap_values": {k: float(v) for k, v in sorted_shap.items()}
        }
    }
    
    with open(output_json_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"\nResults saved to: {output_json_path}")
    print(f"SHAP visualization saved to: {output_path}")

if __name__ == "__main__":
    main() 