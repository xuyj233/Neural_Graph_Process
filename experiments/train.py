"""
Main Training Script for Dynamic Causal Graph Modeling

This script provides the main entry point for training the dynamic causal graph model
with all the implemented loss functions and training objectives.
"""

import argparse
import torch
import numpy as np
import random
import os
import sys
from typing import Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, get_default_config, get_small_config, get_large_config, get_debug_config
from models.integrated_model import IntegratedDynamicCausalModel
from data.data_loader import DataLoaderManager
from training.trainer import Trainer
from utils.simulation import DynamicLSEMSimulator, SimulationConfig, create_simulation_configs
from utils.metrics import MetricsCalculator


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(config: Config) -> IntegratedDynamicCausalModel:
    """Create the integrated dynamic causal model."""
    model = IntegratedDynamicCausalModel(
        d=config.model.d,
        graph_hidden_dim=config.model.graph_hidden_dim,
        obs_hidden_dim=config.model.obs_hidden_dim,
        message_dim=config.model.message_dim,
        num_components=config.model.num_components,
        graph_type=config.model.graph_type,
        discretization_fn=config.model.discretization_fn,
        noise_type=config.model.noise_type,
        encoder_type=config.model.encoder_type
    )
    
    return model


def create_data_loaders(config: Config, device: torch.device, simulation_config: Optional[SimulationConfig] = None):
    """Create data loaders for training."""
    data_manager = DataLoaderManager(config.data, device, simulation_config)
    data_manager.create_dataset(config.model.d)
    data_manager.create_loaders()
    
    train_loader, val_loader, test_loader = data_manager.get_loaders()
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Dynamic Causal Graph Model')
    
    # Configuration arguments
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--config_type', type=str, default='default', 
                       choices=['default', 'small', 'large', 'debug'],
                       help='Type of predefined configuration')
    
    # Model arguments
    parser.add_argument('--d', type=int, default=None, help='Number of variables')
    parser.add_argument('--graph_type', type=str, default=None, choices=['ode', 'sde'],
                       help='Graph model type')
    parser.add_argument('--encoder_type', type=str, default=None, choices=['rnn', 'gru', 'lstm'],
                       help='Encoder type')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    
    # Data arguments
    parser.add_argument('--num_trajectories', type=int, default=None, help='Number of trajectories')
    parser.add_argument('--seq_len', type=int, default=None, help='Sequence length')
    
    # Simulation arguments
    parser.add_argument('--simulation_config', type=str, default=None, 
                       choices=['S1_F1', 'S1_F2', 'S2_F1', 'S2_F2'],
                       help='Predefined simulation configuration')
    parser.add_argument('--scenario', type=str, default=None, choices=['S1', 'S2'],
                       help='Graph scenario')
    parser.add_argument('--function_type', type=str, default=None, choices=['F1', 'F2'],
                       help='Causal function type')
    
    # Experiment arguments
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--run_id', type=str, default=None, help='Run ID')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    # Weights & Biases arguments
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity (username or team)')
    parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='W&B tags')
    parser.add_argument('--wandb_notes', type=str, default=None, help='W&B notes')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--test_only', action='store_true', help='Test only mode')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.load(args.config)
    else:
        if args.config_type == 'default':
            config = get_default_config()
        elif args.config_type == 'small':
            config = get_small_config()
        elif args.config_type == 'large':
            config = get_large_config()
        elif args.config_type == 'debug':
            config = get_debug_config()
        else:
            raise ValueError(f"Unknown config type: {args.config_type}")
    
    # Override configuration with command line arguments
    if args.d is not None:
        config.model.d = args.d
    if args.graph_type is not None:
        config.model.graph_type = args.graph_type
    if args.encoder_type is not None:
        config.model.encoder_type = args.encoder_type
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.num_trajectories is not None:
        config.data.num_trajectories = args.num_trajectories
    if args.seq_len is not None:
        config.data.seq_len = args.seq_len
    if args.experiment_name is not None:
        config.experiment.experiment_name = args.experiment_name
    if args.run_id is not None:
        config.experiment.run_id = args.run_id
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.device != 'auto':
        config.training.device = args.device
    
    # Handle wandb configuration
    if args.use_wandb:
        config.experiment.use_wandb = True
    elif args.no_wandb:
        config.experiment.use_wandb = False
    
    if args.wandb_project is not None:
        config.experiment.wandb_project = args.wandb_project
    if args.wandb_entity is not None:
        config.experiment.wandb_entity = args.wandb_entity
    if args.wandb_tags is not None:
        config.experiment.wandb_tags = args.wandb_tags
    if args.wandb_notes is not None:
        config.experiment.wandb_notes = args.wandb_notes
    
    # Set device
    if config.training.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config.training.device = str(device)
    else:
        device = torch.device(config.training.device)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set seed
    set_seed(config.experiment.seed)
    
    # Create simulation configuration
    simulation_config = None
    if args.simulation_config:
        simulation_configs = create_simulation_configs(config.model.d)
        simulation_config = simulation_configs[args.simulation_config]
    elif args.scenario and args.function_type:
        simulation_config = SimulationConfig(
            scenario=args.scenario,
            function_type=args.function_type,
            p=config.model.d,  # Use model's d parameter
            m=config.data.seq_len,
            T=10,
            n_realizations=config.data.num_trajectories
        )
    
    # Create directories
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    os.makedirs(config.experiment.save_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(config.experiment.save_dir, 'config.json')
    config.save(config_path)
    
    print(f"Configuration saved to: {config_path}")
    print(f"Using device: {device}")
    print(f"Model dimensions: d={config.model.d}")
    print(f"Graph type: {config.model.graph_type}")
    print(f"Encoder type: {config.model.encoder_type}")
    
    # Create model
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config, device, simulation_config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Print simulation info if using Dynamic LSEM
    if simulation_config:
        print(f"Simulation: {simulation_config.scenario} + {simulation_config.function_type}")
        print(f"Variables: {simulation_config.p}, Observations: {simulation_config.m}, Time points: {simulation_config.T}")
    
    # Create trainer
    trainer = Trainer(config, model)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")
    
    # Test only mode
    if args.test_only:
        print("Testing mode - evaluating on test set...")
        test_losses = trainer.validate(test_loader)
        print("Test Results:")
        for key, value in test_losses.items():
            print(f"  {key}: {value:.4f}")
        return
    
    # Training
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    
    # Final evaluation
    print("Final evaluation on test set...")
    test_losses = trainer.validate(test_loader)
    print("Final Test Results:")
    for key, value in test_losses.items():
        print(f"  {key}: {value:.4f}")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
