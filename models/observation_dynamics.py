"""
Observation Dynamics Transformation

This module implements the transformation from causal graph to multivariate time series distribution
using mixture Student-t distributions. It includes time series encoding, message aggregation,
and mixture distribution modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, Optional, Union, List, Dict


class TimeSeriesEncoder(nn.Module):
    """Time series encoder for encoding historical trajectory X_≤t into latent representations H(t)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64, 
                 encoder_type: str = "rnn"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.encoder_type = encoder_type
        
        if encoder_type == "rnn":
            self.encoder = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif encoder_type == "gru":
            self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        elif encoder_type == "lstm":
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, X_history: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Encode historical trajectory into latent representations.
        
        Args:
            X_history: Historical observations of shape (batch_size, seq_len, d)
            timestamps: Timestamps of shape (batch_size, seq_len)
            
        Returns:
            Latent representations H(t) of shape (batch_size, d, q)
        """
        batch_size, seq_len, d = X_history.shape
        
        # Encode each variable independently
        H_list = []
        for i in range(d):
            # Extract time series for variable i
            x_i = X_history[:, :, i:i+1]  # (batch_size, seq_len, 1)
            
            # Encode
            if self.encoder_type == "lstm":
                output, (hidden, cell) = self.encoder(x_i)
            else:
                output, hidden = self.encoder(x_i)
            
            # Use the last hidden state
            h_i = hidden[-1]  # (batch_size, hidden_dim)
            h_i = self.output_projection(h_i)  # (batch_size, output_dim)
            H_list.append(h_i)
        
        # Stack to get H(t) of shape (batch_size, d, q)
        H = torch.stack(H_list, dim=1)  # (batch_size, d, output_dim)
        
        return H


class MessageAggregator(nn.Module):
    """MLP Φ for transforming graph-structured aggregated messages."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, aggregated_messages: torch.Tensor) -> torch.Tensor:
        """
        Transform aggregated messages M(t) = Φ(G(t)H(t)).
        
        Args:
            aggregated_messages: G(t)H(t) of shape (batch_size, d, input_dim)
            
        Returns:
            Transformed messages M(t) of shape (batch_size, d, output_dim)
        """
        batch_size, d, input_dim = aggregated_messages.shape
        
        # Reshape for processing
        messages_flat = aggregated_messages.view(-1, input_dim)  # (batch_size * d, input_dim)
        transformed_flat = self.network(messages_flat)  # (batch_size * d, output_dim)
        
        # Reshape back
        M = transformed_flat.view(batch_size, d, self.output_dim)  # (batch_size, d, output_dim)
        
        return M


class MixtureStudentTHead(nn.Module):
    """Head network Γ_ϑ for predicting mixture Student-t distribution parameters."""
    
    def __init__(self, context_dim: int, num_components: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.context_dim = context_dim
        self.num_components = num_components
        self.hidden_dim = hidden_dim
        
        # Shared network
        self.shared_net = nn.Sequential(
            nn.Linear(context_dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Component-specific heads
        self.pi_head = nn.Linear(hidden_dim, num_components)  # mixture weights
        self.mu_head = nn.Linear(hidden_dim, num_components)  # locations
        self.sigma_head = nn.Linear(hidden_dim, num_components)  # scales
        self.nu_head = nn.Linear(hidden_dim, num_components)  # degrees of freedom
    
    def forward(self, context: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict mixture Student-t distribution parameters.
        
        Args:
            context: Context vector C_i(t) of shape (batch_size, d, context_dim)
            t: Time tensor of shape (batch_size,)
            
        Returns:
            Dictionary containing mixture parameters for each component
        """
        batch_size, d, context_dim = context.shape
        
        # Add time information
        t_expanded = t.unsqueeze(-1).unsqueeze(-1).expand(batch_size, d, 1)
        context_with_time = torch.cat([context, t_expanded], dim=-1)
        
        # Reshape for processing
        context_flat = context_with_time.view(-1, context_dim + 1)
        features = self.shared_net(context_flat)
        
        # Predict parameters
        pi_logits = self.pi_head(features)  # (batch_size * d, num_components)
        mu = self.mu_head(features)  # (batch_size * d, num_components)
        sigma = F.softplus(self.sigma_head(features)) + 1e-6  # Ensure positive
        nu = F.softplus(self.nu_head(features)) + 2.0  # Ensure > 2 for finite variance
        
        # Reshape back
        pi_logits = pi_logits.view(batch_size, d, self.num_components)
        mu = mu.view(batch_size, d, self.num_components)
        sigma = sigma.view(batch_size, d, self.num_components)
        nu = nu.view(batch_size, d, self.num_components)
        
        # Convert logits to probabilities
        pi = F.softmax(pi_logits, dim=-1)
        
        return {
            'pi': pi,      # mixture weights
            'mu': mu,      # locations
            'sigma': sigma, # scales
            'nu': nu       # degrees of freedom
        }


class StudentTDistribution:
    """Student-t distribution with reparameterization for gradient-based optimization."""
    
    @staticmethod
    def sample(mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """
        Sample from Student-t distribution using reparameterization.
        
        Args:
            mu: Location parameter of shape (..., K)
            sigma: Scale parameter of shape (..., K)
            nu: Degrees of freedom of shape (..., K)
            
        Returns:
            Samples of shape (..., K)
        """
        # Sample auxiliary variables
        z = torch.randn_like(mu)  # z ~ N(0, 1)
        u = torch.distributions.Chi2(nu).sample()  # u ~ χ²(ν)
        
        # Reparameterization: x = μ + σ * z / sqrt(u/ν)
        x = mu + sigma * z / torch.sqrt(u / nu)
        
        return x
    
    @staticmethod
    def log_prob(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of Student-t distribution.
        
        Args:
            x: Input values of shape (..., K)
            mu: Location parameter of shape (..., K)
            sigma: Scale parameter of shape (..., K)
            nu: Degrees of freedom of shape (..., K)
            
        Returns:
            Log probabilities of shape (..., K)
        """
        # Student-t log probability
        diff = (x - mu) / sigma
        log_prob = (
            torch.lgamma((nu + 1) / 2) - 
            torch.lgamma(nu / 2) - 
            0.5 * torch.log(nu * math.pi) - 
            torch.log(sigma) - 
            (nu + 1) / 2 * torch.log(1 + diff**2 / nu)
        )
        
        return log_prob


class ObservationDynamicsTransformation(nn.Module):
    """Complete observation dynamics transformation from causal graph to time series distribution."""
    
    def __init__(self, d: int, input_dim: int = 1, hidden_dim: int = 64, 
                 message_dim: int = 32, num_components: int = 3, 
                 encoder_type: str = "gru"):
        super().__init__()
        self.d = d
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.num_components = num_components
        
        # Components
        self.encoder = TimeSeriesEncoder(input_dim, hidden_dim, hidden_dim, encoder_type)
        self.message_aggregator = MessageAggregator(hidden_dim, message_dim)
        self.mixture_head = MixtureStudentTHead(hidden_dim + message_dim, num_components)
        
        # Student-t distribution
        self.student_t = StudentTDistribution()
    
    def forward(self, X_history: torch.Tensor, timestamps: torch.Tensor, 
                G: torch.Tensor, t: torch.Tensor, return_params: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Transform causal graph G(t) to multivariate time series distribution at t+Δt.
        
        Args:
            X_history: Historical observations of shape (batch_size, seq_len, d)
            timestamps: Historical timestamps of shape (batch_size, seq_len)
            G: Causal graph G(t) of shape (batch_size, d, d)
            t: Current time of shape (batch_size,)
            return_params: Whether to return distribution parameters
            
        Returns:
            If return_params=False: Samples X(t+Δt) of shape (batch_size, d)
            If return_params=True: Dictionary with samples and parameters
        """
        batch_size = X_history.shape[0]
        
        # Step 1: Encode historical trajectory H(t) = Enc_ψ({(τ, X(τ)): τ ≤ t})
        H = self.encoder(X_history, timestamps)  # (batch_size, d, hidden_dim)
        
        # Step 2: Graph-structured message aggregation M(t) = Φ(G(t)H(t))
        # G(t)H(t) implements: (G(t)H(t))_i = Σ_{j≠i} G_{ij}(t) H_j(t)
        aggregated_messages = torch.bmm(G, H)  # (batch_size, d, hidden_dim)
        M = self.message_aggregator(aggregated_messages)  # (batch_size, d, message_dim)
        
        # Step 3: Construct context vectors C_i(t) = concat(H_i(t), M_i(t))
        C = torch.cat([H, M], dim=-1)  # (batch_size, d, hidden_dim + message_dim)
        
        # Step 4: Predict mixture parameters {π_{i,k}(t), μ_{i,k}(t), σ_{i,k}(t), ν_{i,k}(t)}
        mixture_params = self.mixture_head(C, t)
        
        # Step 5: Sample from mixture Student-t distribution
        X_next = self._sample_mixture(mixture_params)  # (batch_size, d)
        
        if return_params:
            return {
                'samples': X_next,
                'mixture_params': mixture_params,
                'H': H,
                'M': M,
                'C': C
            }
        else:
            return X_next
    
    def _sample_mixture(self, mixture_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sample from mixture Student-t distribution."""
        pi = mixture_params['pi']  # (batch_size, d, num_components)
        mu = mixture_params['mu']
        sigma = mixture_params['sigma']
        nu = mixture_params['nu']
        
        batch_size, d, num_components = pi.shape
        
        # Sample component indices k ~ Cat(π_i(t))
        component_indices = torch.multinomial(pi.view(-1, num_components), 1)  # (batch_size * d, 1)
        component_indices = component_indices.view(batch_size, d)  # (batch_size, d)
        
        # Sample from selected components
        X_next = torch.zeros(batch_size, d, device=pi.device)
        
        for i in range(d):
            for k in range(num_components):
                mask = (component_indices[:, i] == k)
                if mask.any():
                    # Sample from Student-t distribution
                    samples = self.student_t.sample(
                        mu[mask, i, k:k+1], 
                        sigma[mask, i, k:k+1], 
                        nu[mask, i, k:k+1]
                    )
                    X_next[mask, i] = samples.squeeze(-1)
        
        return X_next
    
    def log_likelihood(self, X_next: torch.Tensor, mixture_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute log likelihood of observations under the mixture Student-t distribution."""
        pi = mixture_params['pi']  # (batch_size, d, num_components)
        mu = mixture_params['mu']
        sigma = mixture_params['sigma']
        nu = mixture_params['nu']
        
        batch_size, d, num_components = pi.shape
        
        # Compute log probabilities for each component
        log_probs = torch.zeros(batch_size, d, num_components, device=X_next.device)
        
        for k in range(num_components):
            log_probs[:, :, k] = self.student_t.log_prob(
                X_next.unsqueeze(-1), 
                mu[:, :, k:k+1], 
                sigma[:, :, k:k+1], 
                nu[:, :, k:k+1]
            ).squeeze(-1)
        
        # Log-sum-exp for mixture: log Σ_k π_k * p_k(x) = logsumexp(log π_k + log p_k(x))
        log_mixture_probs = torch.logsumexp(torch.log(pi + 1e-8) + log_probs, dim=-1)
        
        return log_mixture_probs

