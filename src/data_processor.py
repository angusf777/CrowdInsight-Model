#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data processing utilities for the Kickstarter project success prediction model.

This module handles data loading, preprocessing, and preparation of Kickstarter
project data for model training. It includes functionality for loading JSON data,
splitting it into training/validation/test sets, normalizing numerical features,
and creating PyTorch DataLoader objects. The processor ensures data is properly
formatted for input to the neural network model.

Author: Angus Fung
Date: April 2025
"""

import json
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List, Any
import torch
from torch.utils.data import Dataset, DataLoader

class KickstarterDataProcessor:
    """Kickstarter Project Data Processing Class"""
    
    def __init__(self, data_path: str):
        """
        Initialize data processor
        
        Args:
            data_path: JSON data file path
        """
        self.data_path = data_path
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_data(self) -> None:
        """Load JSON data"""
        print(f"Loading data file: {self.data_path}")
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        print(f"Successfully loaded {len(self.data)} records")
        
    def prepare_features(self, record: Dict) -> Dict[str, Any]:
        """
        Prepare features for a single record
        
        Args:
            record: Single record data
            
        Returns:
            Dictionary containing all features
        """
        # Extract embedding features
        features = {
            'description_embedding': np.array(record.get('description_embedding', [])),
            'blurb_embedding': np.array(record.get('blurb_embedding', [])) if 'blurb_embedding' in record else np.array([]),
            'risk_embedding': np.array(record.get('risk_embedding', [])) if 'risk_embedding' in record else np.array([]),
            'subcategory_embedding': np.array(record.get('subcategory_embedding', [])) if 'subcategory_embedding' in record else np.array([]),
            'category_embedding': np.array(record.get('category_embedding', [])) if 'category_embedding' in record else np.array([]),
            'country_embedding': np.array(record.get('country_embedding', [])) if 'country_embedding' in record else np.array([]),
        }
        
        # Extract numerical features
        numerical_features = [
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
        
        for feature in numerical_features:
            features[feature] = float(record.get(feature, 0))
        
        # Target variable
        features['state'] = int(record.get('state', 0))
        
        return features
    
    def _split_files_exist(self, output_dir: str = "data") -> bool:
        """
        Check if split data files already exist
        
        Args:
            output_dir: Directory to check for split files (default: 'data')
            
        Returns:
            True if all split files exist, False otherwise
        """
        train_path = os.path.join(output_dir, "train_data.json")
        val_path = os.path.join(output_dir, "val_data.json")
        test_path = os.path.join(output_dir, "test_data.json")
        
        return os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)
    
    def _load_split_files(self, output_dir: str = "data") -> bool:
        """
        Load existing split files
        
        Args:
            output_dir: Directory containing split files (default: 'data')
            
        Returns:
            True if files were successfully loaded, False otherwise
        """
        try:
            train_path = os.path.join(output_dir, "train_data.json")
            val_path = os.path.join(output_dir, "val_data.json")
            test_path = os.path.join(output_dir, "test_data.json")
            
            with open(train_path, 'r') as f:
                self.train_data = json.load(f)
                print(f"Loaded training data ({len(self.train_data)} records) from {train_path}")
            
            with open(val_path, 'r') as f:
                self.val_data = json.load(f)
                print(f"Loaded validation data ({len(self.val_data)} records) from {val_path}")
            
            with open(test_path, 'r') as f:
                self.test_data = json.load(f)
                print(f"Loaded test data ({len(self.test_data)} records) from {test_path}")
            
            # Print proportions
            total_len = len(self.train_data) + len(self.val_data) + len(self.test_data)
            train_len = len(self.train_data)
            val_len = len(self.val_data)
            test_len = len(self.test_data)
            
            print(f"Data loading complete: Training set {train_len} ({train_len/total_len*100:.1f}%), "
                  f"Validation set {val_len} ({val_len/total_len*100:.1f}%), "
                  f"Test set {test_len} ({test_len/total_len*100:.1f}%)")
            
            return True
        except Exception as e:
            print(f"Error loading split files: {e}")
            return False
    
    def split_data(self, test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42, output_dir: str = "data") -> None:
        """
        Split data into training, validation, and test sets
        
        Args:
            test_size: Test set proportion (default 0.1 = 10%)
            val_size: Validation set proportion (default 0.1 = 10%)
            random_state: Random seed
            output_dir: Directory to save/load split files (default: 'data')
        """
        if self.data is None:
            raise ValueError("Please load data first")
        
        # Check if split files already exist
        if self._split_files_exist(output_dir):
            print("Split data files already exist, loading them instead of re-splitting")
            if self._load_split_files(output_dir):
                return
            else:
                print("Failed to load existing split files, proceeding with new split")
        
        # Prepare features for all records
        processed_data = [self.prepare_features(record) for record in self.data]
        
        # Split into train+validation and test sets
        train_val_data, self.test_data = train_test_split(
            processed_data, test_size=test_size, random_state=random_state, stratify=[d['state'] for d in processed_data]
        )
        
        # Further split into training and validation sets (ensure validation set is 10% of total data)
        val_ratio = val_size / (1 - test_size)
        self.train_data, self.val_data = train_test_split(
            train_val_data, test_size=val_ratio, random_state=random_state, stratify=[d['state'] for d in train_val_data]
        )
        
        # Print final proportions
        total_len = len(self.data)
        train_len = len(self.train_data)
        val_len = len(self.val_data)
        test_len = len(self.test_data)
        
        print(f"Data split complete: Training set {train_len} ({train_len/total_len*100:.1f}%), "
              f"Validation set {val_len} ({val_len/total_len*100:.1f}%), "
              f"Test set {test_len} ({test_len/total_len*100:.1f}%)")
        
        # Confirm proportions are close to 80-10-10
        if not (0.78 < train_len/total_len < 0.82 and \
                0.08 < val_len/total_len < 0.12 and \
                0.10 < test_len/total_len < 0.12):
            raise ValueError("Data split proportions do not match 80-10-10")
        
        # Save splits to disk
        self.save_splits(output_dir)
    
    def save_splits(self, output_dir: str = "data") -> None:
        """
        Save data splits to disk
        
        Args:
            output_dir: Directory to save splits (default: 'data')
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Helper function to convert numpy arrays back to lists for JSON serialization
        def convert_for_json(data_list):
            converted_data = []
            for record in data_list:
                converted_record = {}
                for key, value in record.items():
                    if isinstance(value, np.ndarray):
                        converted_record[key] = value.tolist()
                    else:
                        converted_record[key] = value
                converted_data.append(converted_record)
            return converted_data
        
        # Save train data
        if self.train_data:
            with open(os.path.join(output_dir, "train_data.json"), 'w') as f:
                json.dump(convert_for_json(self.train_data), f)
            print(f"Saved training data ({len(self.train_data)} records) to {os.path.join(output_dir, 'train_data.json')}")
        
        # Save validation data
        if self.val_data:
            with open(os.path.join(output_dir, "val_data.json"), 'w') as f:
                json.dump(convert_for_json(self.val_data), f)
            print(f"Saved validation data ({len(self.val_data)} records) to {os.path.join(output_dir, 'val_data.json')}")
        
        # Save test data
        if self.test_data:
            with open(os.path.join(output_dir, "test_data.json"), 'w') as f:
                json.dump(convert_for_json(self.test_data), f)
            print(f"Saved test data ({len(self.test_data)} records) to {os.path.join(output_dir, 'test_data.json')}")
    
    def get_dataloaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get PyTorch DataLoaders
        
        Args:
            batch_size: Batch size
            
        Returns:
            DataLoaders for training, validation, and test sets
        """
        train_dataset = KickstarterDataset(self.train_data)
        val_dataset = KickstarterDataset(self.val_data)
        test_dataset = KickstarterDataset(self.test_data)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

    def get_test_dataloader(self, batch_size=32):
        """
        Return all data as test set
        """
        dataset = KickstarterDataset(self.data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class KickstarterDataset(Dataset):
    """Kickstarter Dataset Class"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        """
        Initialize dataset
        
        Args:
            data: Preprocessed data list
        """
        self.data = data
        
        # Check feature dimensions
        if len(data) > 0:
            self.feature_dimensions = {
                'description_embedding': len(data[0]['description_embedding']),
                'blurb_embedding': len(data[0]['blurb_embedding']) if len(data[0]['blurb_embedding']) > 0 else 384,
                'risk_embedding': len(data[0]['risk_embedding']) if len(data[0]['risk_embedding']) > 0 else 384,
                'subcategory_embedding': len(data[0]['subcategory_embedding']) if len(data[0]['subcategory_embedding']) > 0 else 100,
                'category_embedding': len(data[0]['category_embedding']) if len(data[0]['category_embedding']) > 0 else 15,
                'country_embedding': len(data[0]['country_embedding']) if len(data[0]['country_embedding']) > 0 else 100,
            }
        
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get data item
        
        Args:
            idx: Index
            
        Returns:
            Dictionary containing features and labels
        """
        record = self.data[idx]
        
        # Embedding features
        desc_embedding = torch.tensor(record['description_embedding'], dtype=torch.float32)
        blurb_embedding = torch.tensor(record['blurb_embedding'], dtype=torch.float32) if len(record['blurb_embedding']) > 0 else torch.zeros(384, dtype=torch.float32)
        risk_embedding = torch.tensor(record['risk_embedding'], dtype=torch.float32) if len(record['risk_embedding']) > 0 else torch.zeros(384, dtype=torch.float32)
        subcategory_embedding = torch.tensor(record['subcategory_embedding'], dtype=torch.float32) if len(record['subcategory_embedding']) > 0 else torch.zeros(100, dtype=torch.float32)
        category_embedding = torch.tensor(record['category_embedding'], dtype=torch.float32) if len(record['category_embedding']) > 0 else torch.zeros(15, dtype=torch.float32)
        country_embedding = torch.tensor(record['country_embedding'], dtype=torch.float32) if len(record['country_embedding']) > 0 else torch.zeros(100, dtype=torch.float32)
        
        # Numerical features
        numerical_features = [
            float(record.get('description_length', 0)),
            float(record.get('funding_goal', 0)),
            float(record.get('image_count', 0)),
            float(record.get('video_count', 0)),
            float(record.get('campaign_duration', 0)),
            float(record.get('previous_projects_count', 0)),
            float(record.get('previous_success_rate', 0)),
            float(record.get('previous_pledged', 0)),
            float(record.get('previous_funding_goal', 0))
        ]
        
        # Label (use both 'state' and 'label' for compatibility)
        state = torch.tensor(record['state'], dtype=torch.float32)
        
        return {
            'description_embedding': desc_embedding,
            'blurb_embedding': blurb_embedding,
            'risk_embedding': risk_embedding,
            'subcategory_embedding': subcategory_embedding,
            'category_embedding': category_embedding,
            'country_embedding': country_embedding,
            'numerical_features': torch.tensor(numerical_features, dtype=torch.float32),
            'state': state,
            'label': state  # Add label to be compatible with trainer
        } 