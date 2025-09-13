"""
Training Module for Dynamic Causal Graph Modeling

This module implements the complete training objective including:
- Data likelihood loss
- Intervention consistency loss  
- Counterfactual consistency loss
- Graph regularization (sparsity, smoothness, acyclicity)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import time
from tqdm import tqdm
import wandb

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, TrainingConfig
from models.integrated_model import IntegratedDynamicCausalModel
from data.data_loader import DataLoaderManager, apply_interventions, compute_counterfactual_divergence


class LossCalculator:
    """Calculator for all loss components."""
    
    def __init__(self, config: TrainingConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Loss weights
        self.data_weight = config.data_loss_weight
        self.intervention_weight = config.intervention_loss_weight
        self.counterfactual_weight = config.counterfactual_loss_weight
        self.sparsity_weight = config.sparsity_loss_weight
        self.smoothness_weight = config.smoothness_loss_weight
        self.acyclicity_weight = config.acyclicity_loss_weight
        
        # Regularization parameters
        self.sparsity_lambda = config.sparsity_lambda
        self.smoothness_lambda = config.smoothness_lambda
        self.acyclicity_lambda = config.acyclicity_lambda
    
    def compute_data_likelihood_loss(self, model: IntegratedDynamicCausalModel,
                                   batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute data likelihood loss: L_data = -Σ log p(x_i(t) | X_≤t, G_≤t)
        """
        X = batch_data['X']  # (batch_size, seq_len, d)
        timestamps = batch_data['timestamps']  # (batch_size, seq_len)
        
        batch_size, seq_len, d = X.shape
        
        # Create initial adjacency matrix
        A0 = torch.randn(batch_size, d, d, device=self.device) * 0.01
        
        # Create time points for graph evolution
        t_graph = torch.linspace(0, 1, seq_len, device=self.device)
        
        total_loss = 0.0
        num_observations = 0
        
        for t in range(1, seq_len):  # Start from t=1 to have history
            # Historical data up to time t
            X_history = X[:, :t]  # (batch_size, t, d)
            timestamps_history = timestamps[:, :t]  # (batch_size, t)
            
            # Current observations
            X_current = X[:, t]  # (batch_size, d)
            
            # Get model predictions
            result = model(A0, t_graph[:t+1], X_history, timestamps_history, 
                          training=True, return_graph=True)
            
            # Compute log likelihood
            log_likelihood = model.obs_model.log_likelihood(X_current, result['mixture_params'])
            
            # Accumulate loss (negative log likelihood)
            total_loss += -log_likelihood.sum()
            num_observations += batch_size * d
        
        return total_loss / num_observations if num_observations > 0 else torch.tensor(0.0, device=self.device)
    
    def compute_intervention_loss(self, model: IntegratedDynamicCausalModel,
                                batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute intervention consistency loss: L_int = -Σ log p(x_i(t) | do(x_I(t)), X_≤t, G_≤t)
        """
        X = batch_data['X']
        timestamps = batch_data['timestamps']
        interventions = batch_data['interventions']
        intervention_values = batch_data['intervention_values']
        
        batch_size, seq_len, d = X.shape
        
        # Create initial adjacency matrix
        A0 = torch.randn(batch_size, d, d, device=self.device) * 0.01
        t_graph = torch.linspace(0, 1, seq_len, device=self.device)
        
        total_loss = 0.0
        num_interventions = 0
        
        for t in range(1, seq_len):
            # Check if there are interventions at time t
            intervention_mask = interventions[:, t] > 0  # (batch_size, d)
            
            if intervention_mask.any():
                # Historical data
                X_history = X[:, :t]
                timestamps_history = timestamps[:, :t]
                
                # Apply interventions
                X_intervened = apply_interventions(X[:, t], interventions[:, t], intervention_values[:, t])
                
                # Get model predictions with interventions
                result = model(A0, t_graph[:t+1], X_history, timestamps_history,
                              training=True, return_graph=True)
                
                # Compute log likelihood for intervened variables
                log_likelihood = model.obs_model.log_likelihood(X_intervened, result['mixture_params'])
                
                # Only consider intervened variables
                intervention_loss = -log_likelihood * intervention_mask.float()
                total_loss += intervention_loss.sum()
                num_interventions += intervention_mask.sum().item()
        
        return total_loss / num_interventions if num_interventions > 0 else torch.tensor(0.0, device=self.device)
    
    def compute_counterfactual_loss(self, model: IntegratedDynamicCausalModel,
                                  batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute counterfactual consistency loss: L_cf = D(p_cf, p_int)
        """
        X = batch_data['X']
        timestamps = batch_data['timestamps']
        counterfactuals = batch_data['counterfactuals']
        counterfactual_values = batch_data['counterfactual_values']
        
        batch_size, seq_len, d = X.shape
        
        # Create initial adjacency matrix
        A0 = torch.randn(batch_size, d, d, device=self.device) * 0.01
        t_graph = torch.linspace(0, 1, seq_len, device=self.device)
        
        total_loss = 0.0
        num_counterfactuals = 0
        
        for t in range(1, seq_len):
            # Check if there are counterfactuals at time t
            counterfactual_mask = counterfactuals[:, t] > 0
            
            if counterfactual_mask.any():
                # Historical data
                X_history = X[:, :t]
                timestamps_history = timestamps[:, :t]
                
                # Counterfactual scenario
                X_counterfactual = apply_interventions(X[:, t], counterfactuals[:, t], counterfactual_values[:, t])
                
                # Get counterfactual predictions
                result_cf = model(A0, t_graph[:t+1], X_history, timestamps_history,
                                 training=True, return_graph=True)
                
                # Get interventional predictions (same intervention, different conditioning)
                result_int = model(A0, t_graph[:t+1], X_history, timestamps_history,
                                  training=True, return_graph=True)
                
                # Compute divergence between counterfactual and interventional predictions
                divergence = compute_counterfactual_divergence(
                    result_cf['samples'], result_int['samples'], divergence_type="mse"
                )
                
                total_loss += divergence
                num_counterfactuals += 1
        
        return total_loss / num_counterfactuals if num_counterfactuals > 0 else torch.tensor(0.0, device=self.device)
    
    def compute_sparsity_loss(self, model: IntegratedDynamicCausalModel,
                            batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute sparsity regularization: L_sparse = λ_sparse * ||G(t)||_1
        """
        X = batch_data['X']
        timestamps = batch_data['timestamps']
        
        batch_size, seq_len, d = X.shape
        A0 = torch.randn(batch_size, d, d, device=self.device) * 0.01
        t_graph = torch.linspace(0, 1, seq_len, device=self.device)
        
        # Get graph evolution
        result = model(A0, t_graph, X, timestamps, training=True, return_graph=True)
        G_evolution = result['G_evolution']
        
        # Compute L1 norm of graphs over time
        if G_evolution.dim() == 5:  # SDE case
            G_evolution = G_evolution[0]  # Use first sample
        
        sparsity_loss = torch.mean(torch.abs(G_evolution))
        
        return self.sparsity_lambda * sparsity_loss
    
    def compute_smoothness_loss(self, model: IntegratedDynamicCausalModel,
                              batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute temporal smoothness regularization: L_smooth = λ_smooth * ||G(t+1) - G(t)||_2
        """
        X = batch_data['X']
        timestamps = batch_data['timestamps']
        
        batch_size, seq_len, d = X.shape
        A0 = torch.randn(batch_size, d, d, device=self.device) * 0.01
        t_graph = torch.linspace(0, 1, seq_len, device=self.device)
        
        # Get graph evolution
        result = model(A0, t_graph, X, timestamps, training=True, return_graph=True)
        G_evolution = result['G_evolution']
        
        if G_evolution.dim() == 5:  # SDE case
            G_evolution = G_evolution[0]  # Use first sample
        
        # Compute temporal differences
        G_diff = G_evolution[1:] - G_evolution[:-1]
        smoothness_loss = torch.mean(torch.norm(G_diff, dim=(-2, -1)))
        
        return self.smoothness_lambda * smoothness_loss
    
    def compute_acyclicity_loss(self, model: IntegratedDynamicCausalModel,
                              batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute acyclicity constraint: L_acyclic = λ_acyclic * tr(e^G) - d
        """
        X = batch_data['X']
        timestamps = batch_data['timestamps']
        
        batch_size, seq_len, d = X.shape
        A0 = torch.randn(batch_size, d, d, device=self.device) * 0.01
        t_graph = torch.linspace(0, 1, seq_len, device=self.device)
        
        # Get graph evolution
        result = model(A0, t_graph, X, timestamps, training=True, return_graph=True)
        G_evolution = result['G_evolution']
        
        if G_evolution.dim() == 5:  # SDE case
            G_evolution = G_evolution[0]  # Use first sample
        
        # Compute acyclicity constraint for each time step
        acyclicity_loss = 0.0
        for t in range(G_evolution.shape[0]):
            G_t = G_evolution[t]  # (batch_size, d, d)
            
            # Compute tr(e^G) - d for each sample in batch
            for b in range(batch_size):
                G_b = G_t[b]  # (d, d)
                # Use matrix exponential (approximated)
                G_squared = torch.mm(G_b, G_b)
                G_cubed = torch.mm(G_squared, G_b)
                exp_G = torch.eye(d, device=self.device) + G_b + 0.5 * G_squared + (1/6) * G_cubed
                acyclicity_loss += torch.trace(exp_G) - d
        
        acyclicity_loss = acyclicity_loss / (G_evolution.shape[0] * batch_size)
        
        return self.acyclicity_lambda * acyclicity_loss
    
    def compute_total_loss(self, model: IntegratedDynamicCausalModel,
                          batch_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute total loss with all components."""
        losses = {}
        
        # Data likelihood loss
        losses['data'] = self.compute_data_likelihood_loss(model, batch_data)
        
        # Intervention consistency loss
        losses['intervention'] = self.compute_intervention_loss(model, batch_data)
        
        # Counterfactual consistency loss
        losses['counterfactual'] = self.compute_counterfactual_loss(model, batch_data)
        
        # Graph regularization losses
        losses['sparsity'] = self.compute_sparsity_loss(model, batch_data)
        losses['smoothness'] = self.compute_smoothness_loss(model, batch_data)
        losses['acyclicity'] = self.compute_acyclicity_loss(model, batch_data)
        
        # Total weighted loss
        total_loss = (
            self.data_weight * losses['data'] +
            self.intervention_weight * losses['intervention'] +
            self.counterfactual_weight * losses['counterfactual'] +
            self.sparsity_weight * losses['sparsity'] +
            self.smoothness_weight * losses['smoothness'] +
            self.acyclicity_weight * losses['acyclicity']
        )
        
        losses['total'] = total_loss
        
        return losses


class Trainer:
    """Main trainer class for the dynamic causal graph model."""
    
    def __init__(self, config: Config, model: IntegratedDynamicCausalModel):
        self.config = config
        self.model = model
        self.device = torch.device(config.training.device)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize loss calculator
        self.loss_calculator = LossCalculator(config.training, self.device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize Weights & Biases
        self.wandb_run = None
        if self.config.experiment.use_wandb:
            self._init_wandb()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        
    def _create_optimizer(self):
        """Create optimizer based on configuration."""
        if self.config.training.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )
        elif self.config.training.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        if self.config.training.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs
            )
        elif self.config.training.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.training.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        else:
            return None
    
    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.experiment.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _init_wandb(self):
        """Initialize Weights & Biases."""
        try:
            # Create wandb config
            wandb_config = {
                # Model config
                'model/d': self.config.model.d,
                'model/graph_type': self.config.model.graph_type,
                'model/graph_hidden_dim': self.config.model.graph_hidden_dim,
                'model/obs_hidden_dim': self.config.model.obs_hidden_dim,
                'model/message_dim': self.config.model.message_dim,
                'model/encoder_type': self.config.model.encoder_type,
                'model/num_components': self.config.model.num_components,
                'model/discretization_fn': self.config.model.discretization_fn,
                'model/noise_type': self.config.model.noise_type,
                
                # Training config
                'training/num_epochs': self.config.training.num_epochs,
                'training/learning_rate': self.config.training.learning_rate,
                'training/weight_decay': self.config.training.weight_decay,
                'training/optimizer': self.config.training.optimizer,
                'training/scheduler': self.config.training.scheduler,
                'training/batch_size': self.config.data.batch_size,
                
                # Loss weights
                'loss/data_weight': self.config.training.data_loss_weight,
                'loss/intervention_weight': self.config.training.intervention_loss_weight,
                'loss/counterfactual_weight': self.config.training.counterfactual_loss_weight,
                'loss/sparsity_weight': self.config.training.sparsity_loss_weight,
                'loss/smoothness_weight': self.config.training.smoothness_loss_weight,
                'loss/acyclicity_weight': self.config.training.acyclicity_loss_weight,
                
                # Data config
                'data/seq_len': self.config.data.seq_len,
                'data/num_trajectories': self.config.data.num_trajectories,
                'data/simulation_type': self.config.data.simulation_type,
                
                # Experiment config
                'experiment/seed': self.config.experiment.seed,
                'experiment/run_id': self.config.experiment.run_id,
            }
            
            # Initialize wandb run
            self.wandb_run = wandb.init(
                project=self.config.experiment.wandb_project,
                entity=self.config.experiment.wandb_entity,
                name=f"{self.config.experiment.experiment_name}_{self.config.experiment.run_id}",
                config=wandb_config,
                tags=self.config.experiment.wandb_tags,
                notes=self.config.experiment.wandb_notes,
                reinit=True
            )
            
            # Log model architecture
            self._log_model_architecture()
            
            self.logger.info("Weights & Biases initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
            self.wandb_run = None
    
    def _log_model_architecture(self):
        """Log model architecture to wandb."""
        if self.wandb_run is None:
            return
        
        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            wandb.log({
                'model/total_parameters': total_params,
                'model/trainable_parameters': trainable_params,
                'model/parameter_ratio': trainable_params / total_params if total_params > 0 else 0
            })
            
        except Exception as e:
            self.logger.warning(f"Failed to log model architecture: {e}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0.0, 'data': 0.0, 'intervention': 0.0, 'counterfactual': 0.0,
            'sparsity': 0.0, 'smoothness': 0.0, 'acyclicity': 0.0
        }
        num_batches = 0
        
        # Create progress bar for training
        pbar = tqdm(train_loader, desc="Training", leave=False, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, batch_data in enumerate(pbar):
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute losses
            losses = self.loss_calculator.compute_total_loss(self.model, batch_data)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip
                )
            
            # Update parameters
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'Data': f"{losses['data'].item():.4f}",
                'Sparsity': f"{losses['sparsity'].item():.4f}"
            })
            
            # Log to wandb (iteration-level)
            if self.wandb_run is not None:
                wandb.log({
                    'train/iteration/total_loss': losses['total'].item(),
                    'train/iteration/data_loss': losses['data'].item(),
                    'train/iteration/intervention_loss': losses['intervention'].item(),
                    'train/iteration/counterfactual_loss': losses['counterfactual'].item(),
                    'train/iteration/sparsity_loss': losses['sparsity'].item(),
                    'train/iteration/smoothness_loss': losses['smoothness'].item(),
                    'train/iteration/acyclicity_loss': losses['acyclicity'].item(),
                    'train/iteration/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/iteration/global_step': self.global_step
                }, step=self.global_step)
            
            # Increment global step
            self.global_step += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Total Loss: {losses['total'].item():.4f}"
                )
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = {
            'total': 0.0, 'data': 0.0, 'intervention': 0.0, 'counterfactual': 0.0,
            'sparsity': 0.0, 'smoothness': 0.0, 'acyclicity': 0.0
        }
        num_batches = 0
        
        # Create progress bar for validation
        pbar = tqdm(val_loader, desc="Validation", leave=False,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        with torch.no_grad():
            for batch_data in pbar:
                # Move data to device
                for key in batch_data:
                    if isinstance(batch_data[key], torch.Tensor):
                        batch_data[key] = batch_data[key].to(self.device)
                
                # Compute losses
                losses = self.loss_calculator.compute_total_loss(self.model, batch_data)
                
                # Accumulate losses
                for key in val_losses:
                    val_losses[key] += losses[key].item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'Val Loss': f"{losses['total'].item():.4f}",
                    'Data': f"{losses['data'].item():.4f}",
                    'Sparsity': f"{losses['sparsity'].item():.4f}"
                })
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Create main progress bar for epochs
        epoch_pbar = tqdm(range(self.config.training.num_epochs), 
                         desc="Training Progress", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for epoch in epoch_pbar:
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler is not None:
                if self.config.training.scheduler == "plateau":
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Logging
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {train_losses['total']:.4f}, "
                f"Val Loss: {val_losses['total']:.4f}, Time: {epoch_time:.2f}s"
            )
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f"{train_losses['total']:.4f}",
                'Val Loss': f"{val_losses['total']:.4f}",
                'Best': f"{self.best_val_loss:.4f}",
                'Time': f"{epoch_time:.1f}s"
            })
            
            # Log to wandb (epoch-level)
            if self.wandb_run is not None:
                # Calculate evaluation metrics
                eval_metrics = self._compute_evaluation_metrics(train_loader, val_loader)
                
                wandb.log({
                    # Training losses
                    'train/epoch/total_loss': train_losses['total'],
                    'train/epoch/data_loss': train_losses['data'],
                    'train/epoch/intervention_loss': train_losses['intervention'],
                    'train/epoch/counterfactual_loss': train_losses['counterfactual'],
                    'train/epoch/sparsity_loss': train_losses['sparsity'],
                    'train/epoch/smoothness_loss': train_losses['smoothness'],
                    'train/epoch/acyclicity_loss': train_losses['acyclicity'],
                    
                    # Validation losses
                    'val/epoch/total_loss': val_losses['total'],
                    'val/epoch/data_loss': val_losses['data'],
                    'val/epoch/intervention_loss': val_losses['intervention'],
                    'val/epoch/counterfactual_loss': val_losses['counterfactual'],
                    'val/epoch/sparsity_loss': val_losses['sparsity'],
                    'val/epoch/smoothness_loss': val_losses['smoothness'],
                    'val/epoch/acyclicity_loss': val_losses['acyclicity'],
                    
                    # Training metrics
                    'train/epoch/best_val_loss': self.best_val_loss,
                    'train/epoch/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch/epoch_time': epoch_time,
                    'train/epoch/patience_counter': self.patience_counter,
                    
                    # Evaluation metrics
                    **eval_metrics
                }, step=epoch)
            
            # Save checkpoint
            if epoch % self.config.training.save_interval == 0:
                self.save_checkpoint(epoch, val_losses['total'])
            
            # Early stopping
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_losses['total'], is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.logger.info("Training completed!")
        
        # Finish wandb run
        if self.wandb_run is not None:
            wandb.finish()
    
    def _compute_evaluation_metrics(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """Compute evaluation metrics for wandb logging."""
        if self.wandb_run is None:
            return {}
        
        try:
            from utils.metrics import MetricsCalculator
            
            metrics_calc = MetricsCalculator(self.device)
            eval_metrics = {}
            
            # Sample a few batches for evaluation
            self.model.eval()
            with torch.no_grad():
                # Get a sample batch from validation set
                for batch_data in val_loader:
                    # Move data to device
                    for key in batch_data:
                        if isinstance(batch_data[key], torch.Tensor):
                            batch_data[key] = batch_data[key].to(self.device)
                    
                    # Get model predictions
                    X = batch_data['X']
                    timestamps = batch_data['timestamps']
                    
                    # Create initial graph
                    A0 = torch.randn(X.shape[0], self.config.model.d, self.config.model.d, device=self.device)
                    t_graph = torch.linspace(0, 1, X.shape[1], device=self.device)
                    
                    # Forward pass
                    result = self.model(A0, t_graph, X, timestamps, training=False)
                    
                    # Get predicted values and graphs
                    X_pred = result['X_pred']
                    G_evolution = result['G_evolution']
                    
                    # Compute prediction metrics
                    pred_metrics = metrics_calc.calculate_prediction_metrics(X, X_pred)
                    eval_metrics.update({
                        'eval/prediction/mse': pred_metrics['MSE'],
                        'eval/prediction/mae': pred_metrics['MAE'],
                        'eval/prediction/mdd': pred_metrics['MDD'],
                        'eval/prediction/correlation': pred_metrics['Correlation']
                    })
                    
                    # Compute graph metrics if true graph is available
                    if 'true_graph' in batch_data:
                        true_graph = batch_data['true_graph']
                        # Use the last time step graph for evaluation
                        pred_graph = G_evolution[-1]  # (batch_size, d, d)
                        
                        # Average across batch
                        avg_pred_graph = torch.mean(pred_graph, dim=0)
                        
                        graph_metrics = metrics_calc.calculate_graph_metrics(true_graph, avg_pred_graph)
                        eval_metrics.update({
                            'eval/graph/tpr': graph_metrics['TPR'],
                            'eval/graph/fdr': graph_metrics['FDR'],
                            'eval/graph/shd': graph_metrics['SHD'],
                            'eval/graph/precision': graph_metrics['Precision'],
                            'eval/graph/f1': graph_metrics['F1']
                        })
                    
                    break  # Only use first batch for efficiency
            
            self.model.train()
            return eval_metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to compute evaluation metrics: {e}")
            return {}
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config.to_dict()
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = f"{self.config.experiment.save_dir}/checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = f"{self.config.experiment.save_dir}/best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
