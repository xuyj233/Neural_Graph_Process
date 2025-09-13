"""
Configuration module for Dynamic Causal Graph Modeling

This module contains all configurable parameters for the model, training, and data loading.
All parameters are organized in dataclasses for easy access and modification.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    
    # Model dimensions
    d: int = 4  # Number of variables
    graph_hidden_dim: int = 64  # Hidden dimension for graph networks
    obs_hidden_dim: int = 64  # Hidden dimension for observation networks
    message_dim: int = 32  # Message dimension for graph aggregation
    
    # Graph model configuration
    graph_type: str = "sde"  # "ode" or "sde"
    discretization_fn: str = "sigmoid"  # "sigmoid" or "hard_concrete"
    noise_type: str = "independent"  # "independent" or "correlated"
    
    # Observation model configuration
    encoder_type: str = "gru"  # "rnn", "gru", or "lstm"
    num_components: int = 3  # Number of mixture components for Student-t distribution
    
    # Network architecture
    num_layers: int = 3  # Number of layers in drift/diffusion networks
    dropout: float = 0.1  # Dropout rate
    
    # Discretization parameters
    sigmoid_threshold: float = 0.5  # Threshold for sigmoid discretization
    sigmoid_temperature: float = 1.0  # Temperature for sigmoid discretization
    hard_concrete_temperature: float = 1.0  # Temperature for hard concrete
    hard_concrete_stretch: float = 0.1  # Stretch for hard concrete


@dataclass
class DataConfig:
    """Configuration for data loading and generation."""
    
    # Data dimensions
    batch_size: int = 120
    seq_len: int = 20  # Length of time series sequences
    num_trajectories: int = 100000  # Number of trajectories for simulation
    
    # Simulation parameters
    simulation_type: str = "synthetic"  # "synthetic" or "real"
    noise_level: float = 0.1  # Noise level for synthetic data
    intervention_prob: float = 0.1  # Probability of interventions
    counterfactual_prob: float = 0.05  # Probability of counterfactuals
    
    # Time parameters
    time_range: tuple = (0.0, 1.0)  # Time range for simulation
    dt: float = 0.05  # Time step size
    
    # Graph structure for simulation
    graph_structure: str = "chain"  # "chain", "star", "random", or "custom"
    sparsity_level: float = 0.3  # Sparsity level for random graphs
    
    # Data augmentation
    augment_data: bool = True  # Whether to augment data
    augmentation_factor: float = 2.0  # Factor for data augmentation
    
    # Data loading
    num_workers: int = 0  # Number of workers for data loading (0 for Windows compatibility)
    pin_memory: bool = True  # Whether to pin memory for faster GPU transfer


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Training parameters
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Loss weights
    data_loss_weight: float = 1.0  # Weight for data likelihood loss
    intervention_loss_weight: float = 0.5  # Weight for intervention consistency loss
    counterfactual_loss_weight: float = 0.3  # Weight for counterfactual consistency loss
    sparsity_loss_weight: float = 0.1  # Weight for sparsity regularization
    smoothness_loss_weight: float = 0.1  # Weight for temporal smoothness
    acyclicity_loss_weight: float = 0.1  # Weight for acyclicity constraint
    
    # Regularization parameters
    sparsity_lambda: float = 0.01  # Sparsity regularization strength
    smoothness_lambda: float = 0.01  # Temporal smoothness regularization strength
    acyclicity_lambda: float = 0.01  # Acyclicity constraint strength
    
    # Optimization
    optimizer: str = "adam"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 10  # Number of warmup epochs
    
    # Validation and checkpointing
    val_interval: int = 5  # Validation interval (epochs)
    save_interval: int = 10  # Model saving interval (epochs)
    early_stopping_patience: int = 20  # Early stopping patience
    
    # Device and parallelization
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 0  # Number of data loading workers (0 for Windows compatibility)
    pin_memory: bool = True  # Pin memory for data loading


@dataclass
class ExperimentConfig:
    """Configuration for experiments and logging."""
    
    # Experiment identification
    experiment_name: str = "dynamic_causal_graph"
    run_id: str = "default"
    description: str = "Dynamic Causal Graph Modeling experiment"
    
    # Logging
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    
    # Visualization
    plot_interval: int = 20  # Plotting interval (epochs)
    save_plots: bool = True  # Whether to save plots
    
    # Weights & Biases
    use_wandb: bool = True  # Whether to use Weights & Biases
    wandb_project: str = "dynamic-causal-graph"  # W&B project name
    wandb_entity: str = None  # W&B entity (username or team)
    wandb_tags: List[str] = field(default_factory=lambda: ["causal-graph", "time-series"])
    wandb_notes: str = ""  # Additional notes for the experiment
    
    # Reproducibility
    seed: int = 42  # Random seed
    deterministic: bool = True  # Whether to use deterministic operations


@dataclass
class Config:
    """Main configuration class combining all sub-configurations."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set device
        if self.training.device == "auto":
            self.training.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set random seed
        torch.manual_seed(self.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.experiment.seed)
        
        # Create directories
        import os
        os.makedirs(self.experiment.log_dir, exist_ok=True)
        os.makedirs(self.experiment.save_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.model = ModelConfig(**config_dict['model'])
        config.data = DataConfig(**config_dict['data'])
        config.training = TrainingConfig(**config_dict['training'])
        config.experiment = ExperimentConfig(**config_dict['experiment'])
        
        return config


# Predefined configurations for common use cases
def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_small_config() -> Config:
    """Get configuration for small-scale experiments."""
    config = Config()
    config.model.d = 3
    config.model.graph_hidden_dim = 32
    config.model.obs_hidden_dim = 32
    config.model.message_dim = 16
    config.data.batch_size = 16
    config.data.seq_len = 10
    config.data.num_trajectories = 100
    config.training.num_epochs = 50
    config.training.learning_rate = 2e-3
    return config


def get_large_config() -> Config:
    """Get configuration for large-scale experiments."""
    config = Config()
    config.model.d = 8
    config.model.graph_hidden_dim = 128
    config.model.obs_hidden_dim = 128
    config.model.message_dim = 64
    config.data.batch_size = 64
    config.data.seq_len = 50
    config.data.num_trajectories = 10000
    config.training.num_epochs = 200
    config.training.learning_rate = 5e-4
    return config


def get_debug_config() -> Config:
    """Get configuration for debugging."""
    config = Config()
    config.model.d = 2
    config.model.graph_hidden_dim = 16
    config.model.obs_hidden_dim = 16
    config.model.message_dim = 8
    config.data.batch_size = 4
    config.data.seq_len = 5
    config.data.num_trajectories = 20
    config.training.num_epochs = 5
    config.training.learning_rate = 1e-2
    config.training.val_interval = 1
    config.training.save_interval = 2
    return config
