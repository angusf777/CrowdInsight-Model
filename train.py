#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train the Kickstarter Project Success Prediction Model.

This script handles the end-to-end training process for the Kickstarter success
prediction model. It processes command-line arguments, loads and prepares data,
initializes the model architecture, trains the model (with optional hyperparameter
optimization), generates SHAP explanations, and saves model checkpoints.

Example usage:
    Basic training:
        python train.py --data_path data/allProcessed.json

    Training with hyperparameter optimization:
        python train.py --data_path data/allProcessed.json --optimize_hyperparams

Author: Angus Fung
Date: April 2025
"""

import os
import argparse
import torch
import json
from datetime import datetime

from src.data_processor import KickstarterDataProcessor
from src.model import KickstarterModel
from src.trainer import KickstarterTrainer
from src.explainer import KickstarterExplainer
from src.huggingface_api import deploy_to_huggingface

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Kickstarter Project Success Prediction Model Training")
    
    # Data parameters
    parser.add_argument("--data_path", type=str, default="data/allProcessed.json", help="Data file path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--test_size", type=float, default=0.1, help="Test set proportion")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation set proportion")
    
    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden layer dimension")
    parser.add_argument("--dropout_rate", type=float, default=0.4, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--early_stop_patience", type=int, default=10, help="Early stopping patience")
    
    # Hyperparameter optimization
    parser.add_argument("--optimize_hyperparams", action="store_true", help="Whether to perform hyperparameter optimization")
    
    # Output parameters
    parser.add_argument("--checkpoint_dir", type=str, default="models", help="Model checkpoint save directory")
    parser.add_argument("--log_dir", type=str, default="logs", help="Log save directory")
    
    # Deployment parameters
    parser.add_argument("--deploy", action="store_true", help="Whether to deploy to Hugging Face")
    parser.add_argument("--repo_id", type=str, default="angusf777/Kickstarterproject1", help="Hugging Face repository ID")
    
    return parser.parse_args()

def hyperparameter_optimization(data_processor, device):
    """Hyperparameter optimization"""
    print("Starting hyperparameter optimization...")
    
    # Get base data loaders
    train_loader, val_loader, test_loader = data_processor.get_dataloaders(batch_size=32)
    
    # Create base model
    model = KickstarterModel()
    
    # Create trainer
    trainer = KickstarterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device
    )
    
    # Define hyperparameter grid
    param_grid = {
        'model_hidden_dim': [256, 512, 1024],
        'model_dropout_rate': [0.4, 0.3, 0.2],
        'train_learning_rate': [0.001, 0.0005, 0.0001],
        'train_weight_decay': [1e-6, 1e-5, 1e-4],
        'train_batch_size': [16, 32, 64]
    }
    
    # Run hyperparameter optimization
    best_params = trainer.hyperparameter_optimization(param_grid, data_processor)
    
    return best_params

def main():
    """Main function"""
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Load data
    data_processor = KickstarterDataProcessor(args.data_path)
    data_processor.load_data()
    data_processor.split_data(test_size=args.test_size, val_size=args.val_size)
    
    # Get data loaders
    train_loader, val_loader, test_loader = data_processor.get_dataloaders(batch_size=args.batch_size)
    
    # Hyperparameter optimization
    if args.optimize_hyperparams:
        best_params = hyperparameter_optimization(data_processor, device)
        
        # Update parameters
        args.hidden_dim = best_params.get('model_hidden_dim', args.hidden_dim)
        args.dropout_rate = best_params.get('model_dropout_rate', args.dropout_rate)
        args.learning_rate = best_params.get('train_learning_rate', args.learning_rate)
        args.weight_decay = best_params.get('train_weight_decay', args.weight_decay)
        args.batch_size = best_params.get('train_batch_size', args.batch_size)
        
        # Get data loaders again with new batch size
        train_loader, val_loader, test_loader = data_processor.get_dataloaders(batch_size=args.batch_size)
    
    # Create model
    model = KickstarterModel(
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout_rate
    )
    
    # Create trainer
    trainer = KickstarterTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        device=device
    )
    
    # Train model
    print("\nStarting model training...")
    trainer.train(num_epochs=args.num_epochs, early_stop_patience=args.early_stop_patience)
    
    # Explain model with SHAP
    print("\nStarting model explanation...")
    try:
        test_shap_values = trainer.explain_with_shap(test_loader)
        
        # Create explainer and generate example explanations
        explainer = KickstarterExplainer(model, device)
        print("\nGenerating example explanations...")
        explainer.batch_explain(test_loader, num_samples=5)
    except Exception as e:
        print(f"\nSHAP explanation generation failed, error message: {e}")
        print("Continuing deployment process...")
    
    # Save training parameters
    params_path = os.path.join(args.checkpoint_dir, "training_params.json")
    with open(params_path, 'w') as f:
        params = vars(args)
        params['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        json.dump(params, f, indent=4)
    
    # Deploy to Hugging Face
    if args.deploy:
        print("\nDeploying model to Hugging Face...")
        model_path = os.path.join(args.checkpoint_dir, "best_model.pth")
        deploy_to_huggingface(model_path, args.repo_id)

if __name__ == "__main__":
    main() 