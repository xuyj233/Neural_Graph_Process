"""
Data Loading Module for Dynamic Causal Graph Modeling

This module provides data loading functionality including simulation of 
multivariate time series with causal structure, interventions, and counterfactuals.
"""

import torch
import numpy as np
import sys
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DataConfig
from utils.simulation import DynamicLSEMSimulator, SimulationConfig


@dataclass
class TrajectoryData:
    """Container for trajectory data."""
    X: torch.Tensor  # Observations: (seq_len, d)
    timestamps: torch.Tensor  # Timestamps: (seq_len,)
    interventions: Optional[torch.Tensor] = None  # Intervention masks: (seq_len, d)
    intervention_values: Optional[torch.Tensor] = None  # Intervention values: (seq_len, d)
    counterfactuals: Optional[torch.Tensor] = None  # Counterfactual masks: (seq_len, d)
    counterfactual_values: Optional[torch.Tensor] = None  # Counterfactual values: (seq_len, d)
    true_graph: Optional[torch.Tensor] = None  # True causal graph: (d, d)


class CausalTimeSeriesDataset(Dataset):
    """Dataset for causal time series data with interventions and counterfactuals."""
    
    def __init__(self, config: DataConfig, device: torch.device, simulation_config: Optional[SimulationConfig] = None):
        self.config = config
        self.device = device
        self.d = config.batch_size  # This will be set by the model config
        self.trajectories = []
        self.simulation_config = simulation_config
        
        # Generate synthetic data
        if config.simulation_type == "synthetic":
            if simulation_config is not None:
                self._generate_dynamic_lsem_data()
            else:
                self._generate_synthetic_data()
        else:
            raise ValueError(f"Unsupported simulation type: {config.simulation_type}")
    
    def _generate_dynamic_lsem_data(self):
        """Generate data using Dynamic LSEM simulator."""
        simulator = DynamicLSEMSimulator(self.simulation_config, self.device)
        
        # Generate all realizations
        realizations = simulator.generate_all_realizations()
        
        # Convert to trajectory format
        for realization in realizations:
            X = realization['X']  # (T, m, p)
            time_points = realization['time_points']  # (T,)
            true_graph = realization['true_graph']  # (p, p)
            
            # Reshape to (T*m, p) for trajectory format
            T, m, p = X.shape
            X_reshaped = X.view(T * m, p)
            
            # Create timestamps
            timestamps = time_points.unsqueeze(0).repeat_interleave(m, dim=0).flatten()
            
            # Generate interventions and counterfactuals
            interventions, intervention_values = self._generate_interventions(T * m, p)
            counterfactuals, counterfactual_values = self._generate_counterfactuals(T * m, p)
            
            trajectory = TrajectoryData(
                X=X_reshaped,
                timestamps=timestamps,
                interventions=interventions,
                intervention_values=intervention_values,
                counterfactuals=counterfactuals,
                counterfactual_values=counterfactual_values,
                true_graph=true_graph
            )
            
            self.trajectories.append(trajectory)
    
    def _generate_synthetic_data(self):
        """Generate synthetic causal time series data (legacy method)."""
        for _ in range(self.config.num_trajectories):
            trajectory = self._generate_single_trajectory()
            self.trajectories.append(trajectory)
    
    def _generate_single_trajectory(self) -> TrajectoryData:
        """Generate a single trajectory with causal structure."""
        seq_len = self.config.seq_len
        d = self.d
        time_range = self.config.time_range
        noise_level = self.config.noise_level
        
        # Generate timestamps
        timestamps = torch.linspace(time_range[0], time_range[1], seq_len, device=self.device)
        
        # Generate true causal graph
        true_graph = self._generate_causal_graph(d)
        
        # Initialize observations
        X = torch.zeros(seq_len, d, device=self.device)
        
        # Generate time series with causal structure
        for t in range(seq_len):
            for i in range(d):
                if t == 0:
                    # Initial values
                    X[t, i] = noise_level * torch.randn(1, device=self.device)
                else:
                    # Causal dependencies
                    causal_effect = 0.0
                    for j in range(d):
                        if true_graph[j, i] > 0:  # j causes i
                            causal_effect += true_graph[j, i] * X[t-1, j]
                    
                    # Add trend and noise
                    trend = 0.1 * t / seq_len
                    X[t, i] = causal_effect + trend + noise_level * torch.randn(1, device=self.device)
        
        # Generate interventions
        interventions, intervention_values = self._generate_interventions(seq_len, d)
        
        # Generate counterfactuals
        counterfactuals, counterfactual_values = self._generate_counterfactuals(seq_len, d)
        
        return TrajectoryData(
            X=X,
            timestamps=timestamps,
            interventions=interventions,
            intervention_values=intervention_values,
            counterfactuals=counterfactuals,
            counterfactual_values=counterfactual_values,
            true_graph=true_graph
        )
    
    def _generate_causal_graph(self, d: int) -> torch.Tensor:
        """Generate a causal graph with specified structure."""
        G = torch.zeros(d, d, device=self.device)
        
        if self.config.graph_structure == "chain":
            # Chain structure: 0 -> 1 -> 2 -> ...
            for i in range(d - 1):
                G[i, i + 1] = 0.8
        elif self.config.graph_structure == "star":
            # Star structure: 0 -> 1, 0 -> 2, 0 -> 3, ...
            for i in range(1, d):
                G[0, i] = 0.8
        elif self.config.graph_structure == "random":
            # Random structure
            for i in range(d):
                for j in range(d):
                    if i != j and random.random() < self.config.sparsity_level:
                        G[i, j] = random.uniform(0.3, 0.9)
        else:
            raise ValueError(f"Unknown graph structure: {self.config.graph_structure}")
        
        return G
    
    def _generate_interventions(self, seq_len: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate intervention masks and values."""
        interventions = torch.zeros(seq_len, d, device=self.device)
        intervention_values = torch.zeros(seq_len, d, device=self.device)
        
        if random.random() < self.config.intervention_prob:
            # Randomly select intervention time and variables
            intervention_time = random.randint(1, seq_len - 1)
            num_interventions = random.randint(1, min(3, d))
            intervention_vars = random.sample(range(d), num_interventions)
            
            for var in intervention_vars:
                interventions[intervention_time, var] = 1.0
                # Set intervention value (e.g., set to 0 or random value)
                intervention_values[intervention_time, var] = random.uniform(-2.0, 2.0)
        
        return interventions, intervention_values
    
    def _generate_counterfactuals(self, seq_len: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate counterfactual masks and values."""
        counterfactuals = torch.zeros(seq_len, d, device=self.device)
        counterfactual_values = torch.zeros(seq_len, d, device=self.device)
        
        if random.random() < self.config.counterfactual_prob:
            # Randomly select counterfactual time and variables
            counterfactual_time = random.randint(1, seq_len - 1)
            num_counterfactuals = random.randint(1, min(2, d))
            counterfactual_vars = random.sample(range(d), num_counterfactuals)
            
            for var in counterfactual_vars:
                counterfactuals[counterfactual_time, var] = 1.0
                # Set counterfactual value (different from intervention)
                counterfactual_values[counterfactual_time, var] = random.uniform(-1.0, 1.0)
        
        return counterfactuals, counterfactual_values
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> TrajectoryData:
        return self.trajectories[idx]
    
    def get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Get a batch of trajectories."""
        batch_data = {
            'X': [],
            'timestamps': [],
            'interventions': [],
            'intervention_values': [],
            'counterfactuals': [],
            'counterfactual_values': [],
            'true_graphs': []
        }
        
        for idx in indices:
            trajectory = self.trajectories[idx]
            batch_data['X'].append(trajectory.X)
            batch_data['timestamps'].append(trajectory.timestamps)
            batch_data['interventions'].append(trajectory.interventions)
            batch_data['intervention_values'].append(trajectory.intervention_values)
            batch_data['counterfactuals'].append(trajectory.counterfactuals)
            batch_data['counterfactual_values'].append(trajectory.counterfactual_values)
            batch_data['true_graphs'].append(trajectory.true_graph)
        
        # Stack tensors
        for key in batch_data:
            batch_data[key] = torch.stack(batch_data[key])
        
        return batch_data


class DataLoaderManager:
    """Manager for data loading with train/validation/test splits."""
    
    def __init__(self, config: DataConfig, device: torch.device, simulation_config: Optional[SimulationConfig] = None):
        self.config = config
        self.device = device
        self.simulation_config = simulation_config
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def create_dataset(self, d: int):
        """Create dataset with specified number of variables."""
        # Update the dataset's d parameter
        if self.dataset is not None:
            self.dataset.d = d
        else:
            self.dataset = CausalTimeSeriesDataset(self.config, self.device, self.simulation_config)
            self.dataset.d = d
    
    def _collate_fn(self, batch_indices):
        """Collate function for DataLoader (class method to avoid pickle issues)."""
        return self.dataset.get_batch(batch_indices)
    
    def create_loaders(self, train_ratio: float = 0.7, val_ratio: float = 0.15):
        """Create train/validation/test data loaders."""
        if self.dataset is None:
            raise ValueError("Dataset not created. Call create_dataset() first.")
        
        total_size = len(self.dataset)
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        # Create indices
        indices = list(range(total_size))
        random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create data loaders with num_workers=0 for Windows compatibility
        # Windows has issues with multiprocessing and local functions
        num_workers = 0 if sys.platform == "win32" else self.config.num_workers
        
        self.train_loader = DataLoader(
            train_indices,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_indices,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory
        )
        
        self.test_loader = DataLoader(
            test_indices,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory
        )
    
    def get_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train/validation/test data loaders."""
        if self.train_loader is None:
            raise ValueError("Data loaders not created. Call create_loaders() first.")
        
        return self.train_loader, self.val_loader, self.test_loader


def create_data_loaders(config: DataConfig, d: int, device: torch.device, 
                       simulation_config: Optional[SimulationConfig] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function to create data loaders."""
    manager = DataLoaderManager(config, device, simulation_config)
    manager.create_dataset(d)
    manager.create_loaders()
    return manager.get_loaders()


# Utility functions for data processing
def apply_interventions(X: torch.Tensor, interventions: torch.Tensor, 
                       intervention_values: torch.Tensor) -> torch.Tensor:
    """Apply interventions to time series data."""
    X_intervened = X.clone()
    intervention_mask = interventions > 0
    X_intervened[intervention_mask] = intervention_values[intervention_mask]
    return X_intervened


def create_intervention_masks(interventions: torch.Tensor) -> torch.Tensor:
    """Create binary masks for interventions."""
    return (interventions > 0).float()


def compute_intervention_likelihood(X: torch.Tensor, interventions: torch.Tensor,
                                  intervention_values: torch.Tensor,
                                  model, A0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute likelihood under interventions."""
    # Apply interventions
    X_intervened = apply_interventions(X, interventions, intervention_values)
    
    # Compute likelihood with intervened data
    # This is a simplified version - in practice, you'd need to modify the model
    # to handle interventional distributions properly
    with torch.no_grad():
        result = model(A0, t, X_intervened, X_intervened, training=False, return_graph=True)
        # Extract relevant parts for intervention likelihood
        # This would need to be implemented based on the specific model structure
        return result['samples']  # Placeholder


def compute_counterfactual_divergence(counterfactual_pred: torch.Tensor,
                                    interventional_pred: torch.Tensor,
                                    divergence_type: str = "kl") -> torch.Tensor:
    """Compute divergence between counterfactual and interventional predictions."""
    if divergence_type == "kl":
        # KL divergence
        return torch.nn.functional.kl_div(
            torch.log_softmax(counterfactual_pred, dim=-1),
            torch.softmax(interventional_pred, dim=-1),
            reduction='batchmean'
        )
    elif divergence_type == "mse":
        # Mean squared error
        return torch.mean((counterfactual_pred - interventional_pred) ** 2)
    else:
        raise ValueError(f"Unknown divergence type: {divergence_type}")
