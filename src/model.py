#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kickstarter project success prediction model architecture implementation.

This module defines the neural network architecture for predicting Kickstarter
project success. The model implements a multimodal architecture that processes
various input types including text embeddings (description, blurb, risk),
category embeddings (subcategory, category, country), and numerical features.
Each input type is processed by a dedicated branch before features are fused
and passed through fully connected layers for final prediction.

Author: Angus Fung
Date: April 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Any

class KickstarterModel(nn.Module):
    """Kickstarter Project Success Prediction Model"""
    
    def __init__(
        self,
        desc_embedding_dim: int = 768,
        blurb_embedding_dim: int = 384,
        risk_embedding_dim: int = 384,
        subcategory_embedding_dim: int = 100,
        category_embedding_dim: int = 15,
        country_embedding_dim: int = 100,
        numerical_features_dim: int = 9,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3
    ):
        """
        Initialize the model
        
        Args:
            desc_embedding_dim: Description embedding vector dimension
            blurb_embedding_dim: Blurb embedding vector dimension
            risk_embedding_dim: Risk embedding vector dimension
            subcategory_embedding_dim: Subcategory embedding vector dimension
            category_embedding_dim: Category embedding vector dimension
            country_embedding_dim: Country embedding vector dimension
            numerical_features_dim: Numerical features dimension
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout rate
        """
        super(KickstarterModel, self).__init__()
        
        # Description embedding processing
        self.desc_fc = nn.Sequential(
            nn.Linear(desc_embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Blurb embedding processing
        self.blurb_fc = nn.Sequential(
            nn.Linear(blurb_embedding_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Risk embedding processing
        self.risk_fc = nn.Sequential(
            nn.Linear(risk_embedding_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Subcategory embedding processing
        self.subcategory_fc = nn.Sequential(
            nn.Linear(subcategory_embedding_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Category embedding processing
        self.category_fc = nn.Sequential(
            nn.Linear(category_embedding_dim, hidden_dim // 8),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Country embedding processing
        self.country_fc = nn.Sequential(
            nn.Linear(country_embedding_dim, hidden_dim // 8),
            nn.BatchNorm1d(hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Numerical features processing
        self.numerical_fc = nn.Sequential(
            nn.Linear(numerical_features_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined features dimension
        concat_dim = (hidden_dim + 
                     hidden_dim // 2 + 
                     hidden_dim // 2 + 
                     hidden_dim // 4 + 
                     hidden_dim // 8 + 
                     hidden_dim // 8 + 
                     hidden_dim // 4)
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim // 2, 1)
        
        # Save input features for SHAP explanation
        self.input_names = [
            'description_embedding',
            'blurb_embedding',
            'risk_embedding',
            'subcategory_embedding',
            'category_embedding',
            'country_embedding',
            'numerical_features'
        ]
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward propagation
        
        Args:
            inputs: Dictionary containing all input features
            
        Returns:
            Prediction probability and intermediate feature representations
        """
        # Process various embeddings
        desc_out = self.desc_fc(inputs['description_embedding'])
        blurb_out = self.blurb_fc(inputs['blurb_embedding'])
        risk_out = self.risk_fc(inputs['risk_embedding'])
        subcategory_out = self.subcategory_fc(inputs['subcategory_embedding'])
        category_out = self.category_fc(inputs['category_embedding'])
        country_out = self.country_fc(inputs['country_embedding'])
        numerical_out = self.numerical_fc(inputs['numerical_features'])
        
        # Concatenate all features
        combined = torch.cat([
            desc_out, 
            blurb_out, 
            risk_out, 
            subcategory_out, 
            category_out,
            country_out,
            numerical_out
        ], dim=1)
        
        # Fully connected layers
        x = self.fc1(combined)
        x = self.fc2(x)
        
        # Output layer
        logits = self.output(x)
        probs = torch.sigmoid(logits)
        
        # Save intermediate feature representations for SHAP explanation
        intermediate_features = {
            'description_embedding': desc_out,
            'blurb_embedding': blurb_out,
            'risk_embedding': risk_out,
            'subcategory_embedding': subcategory_out,
            'category_embedding': category_out,
            'country_embedding': country_out,
            'numerical_features': numerical_out,
            'combined': combined,
            'fc1': x
        }
        
        return probs.squeeze(1), intermediate_features
    
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Prediction function
        
        Args:
            inputs: Dictionary containing all input features
            
        Returns:
            Prediction probability
        """
        self.eval()
        with torch.no_grad():
            probs, _ = self.forward(inputs)
        return probs 