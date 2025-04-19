#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SHAP-based explanation generation for the Kickstarter success prediction model.

This module implements SHAP (SHapley Additive exPlanations) value generation to
explain the predictions of the Kickstarter success prediction model. It provides
utilities for generating feature contribution values that show how each feature
impacts the model's predictions. The explainer also offers visualization capabilities
to help interpret these explanations.

Author: Angus Fung
Date: April 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Dict, List, Any, Union, Tuple
import os
import json
import matplotlib

from src.model import KickstarterModel

class KickstarterExplainer:
    """Kickstarter prediction model explainer"""
    
    def __init__(self, model: KickstarterModel, device: torch.device = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model.
            device: Computation device.
        """
        self.model = model
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Numerical feature names
        self.numerical_feature_names = [
            'description_length',
            'funding_goal',
            'image_count',
            'video_count',
            'campaign_duration',
            'previous_projects_count',
            'previous_success_rate',
            'previous_pledged',
            'previous_funding_goal'
        ]
        
        # All feature names
        self.all_feature_names = (
            ['description_embedding', 'blurb_embedding', 'risk_embedding', 
             'subcategory_embedding', 'category_embedding', 'country_embedding'] + 
            self.numerical_feature_names
        )
        
        # Mapping from embedding feature names to internal names
        self.embedding_map = {
            'description_embedding': 'description_embedding', 
            'blurb_embedding': 'blurb_embedding', 
            'risk_embedding': 'risk_embedding', 
            'subcategory_embedding': 'subcategory_embedding', 
            'category_embedding': 'category_embedding', 
            'country_embedding': 'country_embedding'
        }
    
    def explain_prediction(self, inputs: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """
        Explain a single prediction.
        
        Args:
            inputs: Input features (using English keys).
            
        Returns:
            Predicted probability and SHAP contribution values (using English keys).
        """
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prediction
        with torch.no_grad():
            probs, intermediate_features = self.model(inputs)
            
        # Calculate SHAP values
        shap_values = {}
        baseline = {k: torch.zeros_like(v) for k, v in inputs.items()}
        
        # Predict baseline
        with torch.no_grad():
            baseline_probs, _ = self.model(baseline)
            
        # Calculate SHAP values for embedding features
        for feature_name, embedding_name in self.embedding_map.items():
            if embedding_name in inputs:
                # Create input containing only the current feature
                feature_input = {k: torch.zeros_like(v) for k, v in inputs.items()}
                feature_input[embedding_name] = inputs[embedding_name]
                
                # Prediction
                with torch.no_grad():
                    feature_probs, _ = self.model(feature_input)
                
                # SHAP value is the prediction difference
                shap_values[feature_name] = (feature_probs - baseline_probs).cpu().item()
        
        # Calculate SHAP values for numerical features
        if 'numerical_features' in inputs:
            num_features = inputs['numerical_features'].size(1)
            for i in range(num_features):
                feature_name = self.numerical_feature_names[i]
                
                # Create input containing only the current numerical feature
                num_input = {k: torch.zeros_like(v) for k, v in inputs.items()}
                num_input['numerical_features'] = torch.zeros_like(inputs['numerical_features'])
                num_input['numerical_features'][:, i] = inputs['numerical_features'][:, i]
                
                # Prediction
                with torch.no_grad():
                    num_probs, _ = self.model(num_input)
                
                # SHAP value is the prediction difference
                shap_values[feature_name] = (num_probs - baseline_probs).cpu().item()
        
        # Return prediction probability and SHAP values
        prediction = probs.cpu().item()
        
        return prediction, shap_values
    
    def visualize_shap_values(self, shap_values: Dict[str, float], output_path: str = "shap_values.png"):
        """
        Visualize SHAP values (using English labels).
        
        Args:
            shap_values: SHAP value dictionary (English keys).
            output_path: Output image path.
        """
        features = list(shap_values.keys())
        values = list(shap_values.values())
        
        # Sort by absolute value
        sorted_idx = np.argsort(np.abs(values))
        features = [features[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        colors = ['red' if x < 0 else 'green' for x in values]
        y_pos = range(len(features))
        
        plt.barh(y_pos, values, color=colors)
        plt.yticks(y_pos, features)
        plt.xlabel('SHAP Value (Negative=Decrease Success Probability, Positive=Increase Success Probability)')
        plt.title('Feature Importance (SHAP Values)')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(output_path)
        plt.close()
    
    def save_explanation(self, prediction: float, shap_values: Dict[str, float], output_path: str = "explanation.json"):
        """
        Save explanation results (using English keys).
        
        Args:
            prediction: Prediction probability.
            shap_values: SHAP value dictionary (English keys).
            output_path: Output JSON path.
        """
        # Sort SHAP values
        sorted_shap = dict(sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True))
        
        # Create output data
        output_data = {
            "prediction": prediction,
            "shap_values": sorted_shap
        }
        
        # Save as JSON
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)

    def batch_explain(self, data_loader, num_samples: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Explain multiple samples in batch.
        
        Args:
            data_loader: DataLoader.
            num_samples: Number of samples to explain.
            
        Returns:
            Explanation results dictionary (using English keys).
        """
        results = []
        count = 0
        
        for batch in data_loader:
            for i in range(batch['label'].size(0)):
                if count >= num_samples:
                    break
                
                # Extract single sample
                sample = {k: v[i:i+1] for k, v in batch.items() if k != 'label'}
                label = batch['label'][i].item()
                
                # Explain prediction
                prediction, shap_values = self.explain_prediction(sample)
                
                # Save results
                result = {
                    "sample_id": count,
                    "true_label": label,
                    "prediction": prediction,
                    "shap_values": shap_values
                }
                results.append(result)
                
                # Visualize
                output_dir = "explanations"
                os.makedirs(output_dir, exist_ok=True)
                self.visualize_shap_values(shap_values, os.path.join(output_dir, f"sample_{count}_shap.png"))
                self.save_explanation(prediction, shap_values, os.path.join(output_dir, f"sample_{count}_explanation.json"))
                
                count += 1
            
            if count >= num_samples:
                break
        
        # Save all results
        all_results_path = os.path.join("explanations", "all_explanations.json")
        with open(all_results_path, 'w') as f:
            json.dump({"explanations": results}, f, indent=4)
        
        return {"explanations": results} 