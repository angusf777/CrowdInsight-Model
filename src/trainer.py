#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training implementation for the Kickstarter project success prediction model.

This module implements the training process for the Kickstarter success prediction
model. It handles the training loop, early stopping, model evaluation, hyperparameter
optimization, model checkpointing, and TensorBoard logging. The trainer also provides
functionality for generating SHAP explanations to understand feature importance.

Author: Angus Fung
Date: April 2025
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, List, Any, Optional, Callable
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from tqdm import tqdm
import json
import time
from datetime import datetime
import shap

from src.model import KickstarterModel

class KickstarterTrainer:
    """Kickstarter Project Prediction Model Trainer"""
    
    def __init__(
        self,
        model: KickstarterModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "models",
        log_dir: str = "logs",
        device: Optional[torch.device] = None
    ):
        """
        Initialize the trainer
        
        Args:
            model: Prediction model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            learning_rate: Learning rate
            weight_decay: Weight decay
            checkpoint_dir: Model checkpoint save directory
            log_dir: Log save directory
            device: Training device
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Define loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        
        # Create save directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
        # Setup TensorBoard
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, current_time))
        
        # Training status
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_brier': [],
            'val_log_loss': []
        }
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch in progress_bar:
            # Move data to device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Accumulate loss
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Record to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, epoch)
        
        return avg_loss
    
    def validate(self, epoch: int, loader: DataLoader, phase: str = "val") -> Tuple[float, float, float, float]:
        """
        Validate the model
        
        Args:
            epoch: Current epoch number
            loader: Data loader
            phase: Phase name (val or test)
            
        Returns:
            Average loss, AUC, Brier score and log loss
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_probs = []
        all_labels = []
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [{phase.capitalize()}]")
        
        with torch.no_grad():
            for batch in progress_bar:
                # Move data to device
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'label'}
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs, _ = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Accumulate loss
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Collect predictions and labels
                all_probs.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Calculate average loss
        avg_loss = total_loss / total_samples
        
        # Calculate evaluation metrics
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_probs)
        brier = brier_score_loss(all_labels, all_probs)
        log_loss_value = log_loss(all_labels, all_probs)
        
        # Record to TensorBoard
        self.writer.add_scalar(f'Loss/{phase}', avg_loss, epoch)
        self.writer.add_scalar(f'Metrics/{phase}_auc', auc, epoch)
        self.writer.add_scalar(f'Metrics/{phase}_brier', brier, epoch)
        self.writer.add_scalar(f'Metrics/{phase}_log_loss', log_loss_value, epoch)
        
        print(f"{phase.capitalize()} - Loss: {avg_loss:.4f}, AUC: {auc:.4f}, Brier: {brier:.4f}, Log Loss: {log_loss_value:.4f}")
        
        return avg_loss, auc, brier, log_loss_value
    
    def train(self, num_epochs: int = 50, early_stop_patience: int = 10) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            num_epochs: Number of training epochs
            early_stop_patience: Early stopping patience
            
        Returns:
            Training history
        """
        print(f"Starting training, device: {self.device}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_auc, val_brier, val_log_loss = self.validate(epoch, self.val_loader, "val")
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_auc'].append(val_auc)
            self.train_history['val_brier'].append(val_brier)
            self.train_history['val_log_loss'].append(val_log_loss)
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_epoch = epoch
                self.early_stop_counter = 0
                
                checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_auc,
                    'val_brier': val_brier,
                    'val_log_loss': val_log_loss
                }, checkpoint_path)
                
                print(f"Epoch {epoch+1}: Saved best model, validation AUC: {val_auc:.4f}")
            else:
                self.early_stop_counter += 1
                print(f"Epoch {epoch+1}: Validation AUC did not improve, current best: {self.best_val_auc:.4f} (epoch {self.best_epoch+1})")
            
            # Early stopping
            if self.early_stop_counter >= early_stop_patience:
                print(f"No validation performance improvement for {early_stop_patience} consecutive epochs, early stopping")
                break
        
        # Load best model
        checkpoint_path = os.path.join(self.checkpoint_dir, f"best_model.pth")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model (epoch {checkpoint['epoch']+1}), validation AUC: {checkpoint['val_auc']:.4f}")
        
        # Test best model
        test_loss, test_auc, test_brier, test_log_loss = self.validate(checkpoint['epoch'], self.test_loader, "test")
        
        # Save training history
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            history_data = {
                'train_history': self.train_history,
                'best_epoch': self.best_epoch,
                'best_val_auc': self.best_val_auc,
                'test_metrics': {
                    'loss': test_loss,
                    'auc': test_auc,
                    'brier': test_brier,
                    'log_loss': test_log_loss
                }
            }
            json.dump(history_data, f, indent=4)
        
        print(f"Training completed, best validation AUC: {self.best_val_auc:.4f}, test AUC: {test_auc:.4f}")
        
        return self.train_history
    
    def explain_with_shap(self, data_loader: DataLoader, num_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Explain model using SHAP
        
        Args:
            data_loader: Data loader
            num_samples: Number of samples for calculating SHAP values
            
        Returns:
            Dictionary of feature SHAP values
        """
        print("Starting model explanation using SHAP...")
        
        try:
            # Import the KickstarterExplainer
            from src.explainer import KickstarterExplainer
            
            # Create explainer instance
            explainer = KickstarterExplainer(self.model, self.device)
            
            # Process samples from data loader
            results = explainer.batch_explain(data_loader, num_samples=num_samples)
            
            # Save SHAP values to checkpoint directory
            os.makedirs(os.path.join(self.checkpoint_dir, "explanations"), exist_ok=True)
            
            # Export a summary of SHAP values
            shap_values_summary = {}
            
            # Extract and average SHAP values across samples
            if results["explanations"]:
                # Initialize with first sample's feature names
                for feature, value in results["explanations"][0]["shap_values"].items():
                    shap_values_summary[feature] = []
                
                # Collect values across all samples
                for sample_result in results["explanations"]:
                    for feature, value in sample_result["shap_values"].items():
                        shap_values_summary[feature].append(value)
                
                # Calculate summary statistics
                shap_summary = {}
                for feature, values in shap_values_summary.items():
                    shap_summary[feature] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "abs_mean": float(np.mean(np.abs(values)))
                    }
                
                # Save summary to file
                summary_path = os.path.join(self.checkpoint_dir, "shap_summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(shap_summary, f, indent=4)
                
                print(f"SHAP summary saved to {summary_path}")
                
                # Print top 5 most important features
                sorted_features = sorted(shap_summary.items(), key=lambda x: x[1]["abs_mean"], reverse=True)
                print("\nTop 5 most important features (by absolute SHAP value):")
                for feature, stats in sorted_features[:5]:
                    print(f"  {feature}: {stats['mean']:.6f} ± {stats['std']:.6f}")
                
                return shap_summary
            else:
                print("No explanations were generated")
                return {}
                
        except Exception as e:
            print(f"SHAP explanation generation failed, error message: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def hyperparameter_optimization(self, param_grid: Dict[str, List[Any]], data_processor) -> Dict[str, Any]:
        """
        Hyperparameter optimization
        
        Args:
            param_grid: Hyperparameter grid
            data_processor: Data processor
            
        Returns:
            Best hyperparameters
        """
        print("Starting hyperparameter optimization...")
        
        best_auc = 0.0
        best_params = None
        results = []
        
        # Initialize TensorBoard
        hparam_writer = SummaryWriter(log_dir=os.path.join("logs", "hparam_tuning"))
        
        # Calculate total number of combinations
        total_combinations = 1
        for param_values in param_grid.values():
            total_combinations *= len(param_values)
        
        print(f"Total of {total_combinations} hyperparameter combinations to try")
        
        # Perform grid search
        combination_index = 0
        
        # Get all parameter names and possible values
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        # Recursive function to generate all parameter combinations
        def grid_search(current_params, depth):
            nonlocal best_auc, best_params, combination_index
            
            if depth == len(param_names):
                combination_index += 1
                print(f"\nTrying hyperparameter combination {combination_index}/{total_combinations}:")
                for name, value in current_params.items():
                    print(f"  {name}: {value}")
                
                # Create model with current hyperparameters
                model_params = {k: v for k, v in current_params.items() if k.startswith('model_')}
                model_params = {k[6:]: v for k, v in model_params.items()}  # Remove 'model_' prefix
                
                train_params = {k: v for k, v in current_params.items() if k.startswith('train_')}
                train_params = {k[6:]: v for k, v in train_params.items()}  # Remove 'train_' prefix
                
                # Get data loaders again with new batch_size
                batch_size = train_params.get('batch_size', 32)
                train_loader, val_loader, test_loader = data_processor.get_dataloaders(batch_size=batch_size)
                
                # Create new model
                model = KickstarterModel(**model_params)
                
                # Create temporary trainer
                trainer = KickstarterTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    learning_rate=train_params.get('learning_rate', 0.001),
                    weight_decay=train_params.get('weight_decay', 1e-5),
                    checkpoint_dir=os.path.join("models", f"hparam_run_{combination_index}"),
                    log_dir=os.path.join("logs", f"hparam_run_{combination_index}"),
                    device=self.device
                )
                
                # Train model
                num_epochs = train_params.get('num_epochs', 20)
                early_stop_patience = train_params.get('early_stop_patience', 5)
                
                trainer.train(num_epochs=num_epochs, early_stop_patience=early_stop_patience)
                
                # Evaluate best model
                checkpoint_path = os.path.join(trainer.checkpoint_dir, "best_model.pth")
                if os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path)
                    val_auc = checkpoint['val_auc']
                    
                    # Record results
                    result = {
                        'params': current_params.copy(),
                        'val_auc': val_auc,
                        'best_epoch': checkpoint['epoch']
                    }
                    results.append(result)
                    
                    # Record hyperparameters to TensorBoard
                    hparam_dict = {f"hparam/{k}": v for k, v in current_params.items()}
                    metric_dict = {
                        'hparam/val_auc': val_auc,
                        'hparam/val_brier': checkpoint['val_brier'],
                        'hparam/val_log_loss': checkpoint['val_log_loss']
                    }
                    hparam_writer.add_hparams(hparam_dict, metric_dict)
                    
                    # Update best parameters
                    if val_auc > best_auc:
                        best_auc = val_auc
                        best_params = current_params.copy()
                        print(f"Found new best hyperparameters! Validation AUC: {val_auc:.4f}")
                    
                    # Cleanup
                    del trainer
                    del model
                    torch.cuda.empty_cache()
                
                return
            
            # Recursively try all possible values for current parameter
            param_name = param_names[depth]
            for param_value in param_values[depth]:
                current_params[param_name] = param_value
                grid_search(current_params, depth + 1)
        
        # Start grid search
        grid_search({}, 0)
        
        # Save results
        results_path = os.path.join("models", "hparam_results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'results': results,
                'best_params': best_params,
                'best_val_auc': best_auc
            }, f, indent=4)
        
        print(f"Hyperparameter optimization completed, best validation AUC: {best_auc:.4f}")
        print("Best hyperparameters:")
        for name, value in best_params.items():
            print(f"  {name}: {value}")
        
        hparam_writer.close()
        
        return best_params 