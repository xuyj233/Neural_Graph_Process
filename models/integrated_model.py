"""
Integrated Dynamic Causal Model

This module combines the dynamic causal graph modeling with observation dynamics transformation
to create a complete end-to-end model for causal graph evolution and time series prediction.
"""

import torch
import torch.nn as nn
from typing import Union, Dict
from .dynamic_graph import DynamicCausalGraphODE, DynamicCausalGraphSDE
from .observation_dynamics import ObservationDynamicsTransformation


class IntegratedDynamicCausalModel(nn.Module):
    """Integrated model combining dynamic causal graph and observation dynamics transformation."""
    
    def __init__(self, d: int, graph_hidden_dim: int = 64, obs_hidden_dim: int = 64,
                 message_dim: int = 32, num_components: int = 3, 
                 graph_type: str = "sde", discretization_fn: str = "sigmoid",
                 noise_type: str = "independent", encoder_type: str = "gru"):
        super().__init__()
        self.d = d
        
        # Dynamic causal graph model
        if graph_type.lower() == "ode":
            self.graph_model = DynamicCausalGraphODE(d, graph_hidden_dim, discretization_fn)
        elif graph_type.lower() == "sde":
            self.graph_model = DynamicCausalGraphSDE(d, graph_hidden_dim, discretization_fn, noise_type)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        # Observation dynamics transformation
        self.obs_model = ObservationDynamicsTransformation(
            d, input_dim=1, hidden_dim=obs_hidden_dim, 
            message_dim=message_dim, num_components=num_components, 
            encoder_type=encoder_type
        )
    
    def forward(self, A0: torch.Tensor, t_graph: torch.Tensor, 
                X_history: torch.Tensor, timestamps: torch.Tensor,
                training: bool = True, return_graph: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Complete forward pass: causal graph evolution → time series prediction.
        
        Args:
            A0: Initial adjacency matrix of shape (..., d, d)
            t_graph: Time points for graph evolution of shape (T,)
            X_history: Historical observations of shape (batch_size, seq_len, d)
            timestamps: Historical timestamps of shape (batch_size, seq_len)
            training: Whether in training mode
            return_graph: Whether to return the evolved causal graph
            
        Returns:
            If return_graph=False: Predicted X(t+Δt) of shape (batch_size, d)
            If return_graph=True: Dictionary with predictions and graph evolution
        """
        # Step 1: Evolve causal graph G(t)
        if isinstance(self.graph_model, DynamicCausalGraphSDE):
            G_evolution = self.graph_model(A0, t_graph, training, num_samples=1)
            G_current = G_evolution[0, -1]  # Use last time point, first sample
        else:
            G_evolution = self.graph_model(A0, t_graph, training)
            G_current = G_evolution[-1]  # Use last time point
        
        # Step 2: Transform to time series distribution
        t_current = t_graph[-1].expand(X_history.shape[0])  # Current time for each batch
        
        if return_graph:
            result = self.obs_model(X_history, timestamps, G_current, t_current, return_params=True)
            result['G_evolution'] = G_evolution
            result['G_current'] = G_current
            return result
        else:
            return self.obs_model(X_history, timestamps, G_current, t_current, return_params=False)
    
    def predict_with_uncertainty(self, A0: torch.Tensor, t_graph: torch.Tensor,
                                X_history: torch.Tensor, timestamps: torch.Tensor,
                                num_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Predict with uncertainty quantification using multiple samples.
        
        Args:
            A0: Initial adjacency matrix of shape (..., d, d)
            t_graph: Time points for graph evolution of shape (T,)
            X_history: Historical observations of shape (batch_size, seq_len, d)
            timestamps: Historical timestamps of shape (batch_size, seq_len)
            num_samples: Number of samples for uncertainty quantification
            
        Returns:
            Dictionary containing mean, std, and samples
        """
        samples = []
        
        for _ in range(num_samples):
            if isinstance(self.graph_model, DynamicCausalGraphSDE):
                G_evolution = self.graph_model(A0, t_graph, training=True, num_samples=1)
                G_current = G_evolution[0, -1]
            else:
                G_evolution = self.graph_model(A0, t_graph, training=True)
                G_current = G_evolution[-1]
            
            t_current = t_graph[-1].expand(X_history.shape[0])
            sample = self.obs_model(X_history, timestamps, G_current, t_current, return_params=False)
            samples.append(sample)
        
        # Stack samples
        samples_tensor = torch.stack(samples, dim=0)  # (num_samples, batch_size, d)
        
        # Compute statistics
        mean_pred = samples_tensor.mean(dim=0)
        std_pred = samples_tensor.std(dim=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'samples': samples_tensor
        }
    
    def compute_loss(self, A0: torch.Tensor, t_graph: torch.Tensor,
                    X_history: torch.Tensor, timestamps: torch.Tensor,
                    X_target: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for training.
        
        Args:
            A0: Initial adjacency matrix of shape (..., d, d)
            t_graph: Time points for graph evolution of shape (T,)
            X_history: Historical observations of shape (batch_size, seq_len, d)
            timestamps: Historical timestamps of shape (batch_size, seq_len)
            X_target: Target observations of shape (batch_size, d)
            
        Returns:
            Negative log-likelihood loss
        """
        # Get predictions with parameters
        result = self.forward(A0, t_graph, X_history, timestamps, 
                            training=True, return_graph=True)
        
        # Compute log likelihood
        log_likelihood = self.obs_model.log_likelihood(X_target, result['mixture_params'])
        
        # Return negative log likelihood (loss)
        return -log_likelihood.mean()
