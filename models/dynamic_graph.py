"""
Dynamic Causal Graph Modeling

This module implements the dynamic causal graph modeling algorithm with both ODE and SDE versions.
The algorithm models evolving causal relations among variables using a latent adjacency process A(t)
that evolves according to stochastic differential equations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Union
from .base_components import (
    DiscretizationFunction, ThresholdedSigmoid, HardConcreteSampler,
    DriftNetwork, DiffusionNetwork, create_initial_adjacency
)


class DynamicCausalGraphODE(nn.Module):
    """ODE version of Dynamic Causal Graph Modeling."""
    
    def __init__(self, d: int, hidden_dim: int = 64, discretization_fn: str = "sigmoid"):
        super().__init__()
        self.d = d
        self.drift_net = DriftNetwork(d, hidden_dim)
        
        if discretization_fn == "sigmoid":
            self.discretization = ThresholdedSigmoid()
        elif discretization_fn == "hard_concrete":
            self.discretization = HardConcreteSampler()
        else:
            raise ValueError(f"Unknown discretization function: {discretization_fn}")
    
    def forward(self, A0: torch.Tensor, t: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Solve ODE: dA(t) = f_θ_f(A(t), t) dt
        
        Args:
            A0: Initial adjacency matrix of shape (..., d, d)
            t: Time points of shape (T,)
            training: Whether in training mode
            
        Returns:
            Graph G(t) of shape (T, ..., d, d)
        """
        T = t.shape[0]
        batch_shape = A0.shape[:-2]
        
        # Initialize solution
        A = A0.unsqueeze(0).expand(T, *batch_shape, self.d, self.d).clone()
        
        # Simple Euler integration
        dt = t[1] - t[0] if T > 1 else 1.0
        
        for i in range(1, T):
            drift = self.drift_net(A[i-1], t[i-1])
            A[i] = A[i-1] + drift * dt
        
        # Transform to causal graph G(t) = σ(A(t)) ⊙ (1 - I)
        G = self._transform_to_graph(A, training)
        
        return G
    
    def _transform_to_graph(self, A: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Transform latent weights A(t) to causal graph G(t)."""
        # Apply discretization function
        G = self.discretization(A, training)
        
        # Remove self-loops: G(t) = σ(A(t)) ⊙ (1 - I)
        I = torch.eye(self.d, device=A.device, dtype=A.dtype)
        G = G * (1 - I)
        
        return G


class DynamicCausalGraphSDE(nn.Module):
    """SDE version of Dynamic Causal Graph Modeling."""
    
    def __init__(self, d: int, hidden_dim: int = 64, discretization_fn: str = "sigmoid", 
                 noise_type: str = "independent"):
        super().__init__()
        self.d = d
        self.noise_type = noise_type
        self.drift_net = DriftNetwork(d, hidden_dim)
        self.diffusion_net = DiffusionNetwork(d, hidden_dim)
        
        if discretization_fn == "sigmoid":
            self.discretization = ThresholdedSigmoid()
        elif discretization_fn == "hard_concrete":
            self.discretization = HardConcreteSampler()
        else:
            raise ValueError(f"Unknown discretization function: {discretization_fn}")
    
    def forward(self, A0: torch.Tensor, t: torch.Tensor, training: bool = True, 
                num_samples: int = 1) -> torch.Tensor:
        """
        Solve SDE: dA(t) = f_θ_f(A(t), t) dt + g_θ_g(A(t), t) dW(t)
        
        Args:
            A0: Initial adjacency matrix of shape (..., d, d)
            t: Time points of shape (T,)
            training: Whether in training mode
            num_samples: Number of Monte Carlo samples for SDE
            
        Returns:
            Graph G(t) of shape (num_samples, T, ..., d, d)
        """
        T = t.shape[0]
        batch_shape = A0.shape[:-2]
        
        # Initialize solution
        A = A0.unsqueeze(0).unsqueeze(0).expand(num_samples, T, *batch_shape, self.d, self.d).clone()
        
        # Simple Euler-Maruyama integration
        dt = t[1] - t[0] if T > 1 else 1.0
        
        for i in range(1, T):
            # Compute drift and diffusion terms
            drift = self.drift_net(A[:, i-1], t[i-1])  # (num_samples, ..., d, d)
            diffusion = self.diffusion_net(A[:, i-1], t[i-1])  # (num_samples, ..., d, d)
            
            # Generate Brownian motion increment
            dW = self._generate_brownian_increment(batch_shape, dt)
            
            # Euler-Maruyama step
            A[:, i] = A[:, i-1] + drift * dt + diffusion * dW
        
        # Transform to causal graph G(t) = σ(A(t)) ⊙ (1 - I)
        G = self._transform_to_graph(A, training)
        
        return G
    
    def _generate_brownian_increment(self, batch_shape: Tuple, dt: float) -> torch.Tensor:
        """Generate Brownian motion increment dW(t)."""
        if self.noise_type == "independent":
            # Independent noise for each edge
            return torch.randn(*batch_shape, self.d, self.d) * np.sqrt(dt)
        elif self.noise_type == "correlated":
            # Correlated noise modeling shared latent factors
            # Generate correlated noise using a low-rank structure
            rank = min(3, self.d)  # Use low-rank approximation
            U = torch.randn(*batch_shape, self.d, rank)
            V = torch.randn(*batch_shape, rank, self.d)
            noise = torch.bmm(U.view(-1, self.d, rank), V.view(-1, rank, self.d))
            noise = noise.view(*batch_shape, self.d, self.d)
            return noise * np.sqrt(dt)
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")
    
    def _transform_to_graph(self, A: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Transform latent weights A(t) to causal graph G(t)."""
        # Apply discretization function
        G = self.discretization(A, training)
        
        # Remove self-loops: G(t) = σ(A(t)) ⊙ (1 - I)
        I = torch.eye(self.d, device=A.device, dtype=A.dtype)
        G = G * (1 - I)
        
        return G


# Example usage function
def example_usage():
    """Example usage of the Dynamic Causal Graph models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Parameters
    d = 5  # Number of variables
    T = 10  # Number of time points
    batch_size = 2
    
    # Create time points
    t = torch.linspace(0, 1, T, device=device)
    
    # Create initial adjacency matrix
    A0 = create_initial_adjacency(d, batch_size, "small_random").to(device)
    
    print("=== ODE Version ===")
    ode_model = DynamicCausalGraphODE(d, hidden_dim=32).to(device)
    G_ode = ode_model(A0, t, training=True)
    print(f"ODE Graph shape: {G_ode.shape}")
    print(f"ODE Graph range: [{G_ode.min():.3f}, {G_ode.max():.3f}]")
    
    print("\n=== SDE Version ===")
    sde_model = DynamicCausalGraphSDE(d, hidden_dim=32, noise_type="independent").to(device)
    G_sde = sde_model(A0, t, training=True, num_samples=3)
    print(f"SDE Graph shape: {G_sde.shape}")
    print(f"SDE Graph range: [{G_sde.min():.3f}, {G_sde.max():.3f}]")
    
    # Test evaluation mode (discrete graphs)
    print("\n=== Evaluation Mode (Discrete) ===")
    G_ode_discrete = ode_model(A0, t, training=False)
    G_sde_discrete = sde_model(A0, t, training=False, num_samples=1)
    print(f"ODE Discrete values: {torch.unique(G_ode_discrete)}")
    print(f"SDE Discrete values: {torch.unique(G_sde_discrete)}")


if __name__ == "__main__":
    example_usage()

