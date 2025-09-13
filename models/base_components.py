"""
Base Components for Dynamic Causal Graph Modeling

This module contains the fundamental building blocks including discretization functions,
neural networks for drift and diffusion terms, and utility functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Union
from abc import ABC, abstractmethod


class DiscretizationFunction(nn.Module):
    """Base class for discretization functions σ(·) that map latent weights to [0,1]."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply discretization function.
        
        Args:
            x: Input tensor of shape (..., d, d)
            training: Whether in training mode (continuous relaxation) or evaluation mode (discrete)
            
        Returns:
            Discretized tensor in [0,1] during training or {0,1} during evaluation
        """
        pass


class ThresholdedSigmoid(DiscretizationFunction):
    """Thresholded sigmoid discretization function."""
    
    def __init__(self, threshold: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            # Continuous relaxation using sigmoid
            return torch.sigmoid(x / self.temperature)
        else:
            # Discrete thresholding
            return (torch.sigmoid(x / self.temperature) > self.threshold).float()


class HardConcreteSampler(DiscretizationFunction):
    """Hard concrete sampler for discretization."""
    
    def __init__(self, temperature: float = 1.0, stretch: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.stretch = stretch
    
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            # Hard concrete relaxation
            u = torch.rand_like(x)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + x) / self.temperature)
            s_bar = s * self.stretch + (1 - self.stretch) * 0.5
            return torch.clamp(s_bar, 0, 1)
        else:
            # Hard thresholding
            return (torch.sigmoid(x / self.temperature) > 0.5).float()


class DriftNetwork(nn.Module):
    """Neural network f_θ_f for the deterministic drift term."""
    
    def __init__(self, d: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.d = d
        self.input_dim = d * d + 1  # flattened adjacency matrix + time
        
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, d * d))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, A: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift term f_θ_f(A(t), t).
        
        Args:
            A: Adjacency matrix of shape (..., d, d)
            t: Time tensor of shape (...)
            
        Returns:
            Drift term of shape (..., d, d)
        """
        batch_shape = A.shape[:-2]
        A_flat = A.view(*batch_shape, -1)  # (..., d*d)
        t_expanded = t.unsqueeze(-1).expand(*batch_shape, 1)  # (..., 1)
        
        input_tensor = torch.cat([A_flat, t_expanded], dim=-1)  # (..., d*d+1)
        output_flat = self.network(input_tensor)  # (..., d*d)
        
        return output_flat.view(*batch_shape, self.d, self.d)


class DiffusionNetwork(nn.Module):
    """Neural network g_θ_g for the stochastic diffusion term."""
    
    def __init__(self, d: int, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.d = d
        self.input_dim = d * d + 1  # flattened adjacency matrix + time
        
        layers = []
        layers.append(nn.Linear(self.input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, d * d))
        layers.append(nn.Softplus())  # Ensure positive diffusion
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, A: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion term g_θ_g(A(t), t).
        
        Args:
            A: Adjacency matrix of shape (..., d, d)
            t: Time tensor of shape (...)
            
        Returns:
            Diffusion term of shape (..., d, d)
        """
        batch_shape = A.shape[:-2]
        A_flat = A.view(*batch_shape, -1)  # (..., d*d)
        t_expanded = t.unsqueeze(-1).expand(*batch_shape, 1)  # (..., 1)
        
        input_tensor = torch.cat([A_flat, t_expanded], dim=-1)  # (..., d*d+1)
        output_flat = self.network(input_tensor)  # (..., d*d)
        
        return output_flat.view(*batch_shape, self.d, self.d)


def create_initial_adjacency(d: int, batch_size: Optional[int] = None, 
                           init_type: str = "random") -> torch.Tensor:
    """
    Create initial adjacency matrix A(0).
    
    Args:
        d: Number of variables
        batch_size: Batch size (optional)
        init_type: Initialization type ("random", "zeros", "small_random")
        
    Returns:
        Initial adjacency matrix of shape (batch_size, d, d) or (d, d)
    """
    if batch_size is None:
        shape = (d, d)
    else:
        shape = (batch_size, d, d)
    
    if init_type == "random":
        return torch.randn(shape) * 0.1
    elif init_type == "zeros":
        return torch.zeros(shape)
    elif init_type == "small_random":
        return torch.randn(shape) * 0.01
    else:
        raise ValueError(f"Unknown initialization type: {init_type}")

