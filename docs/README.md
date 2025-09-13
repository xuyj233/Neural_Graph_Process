# Dynamic Causal Graph Modeling

A complete implementation of Dynamic Causal Graph Modeling with training pipeline, supporting both ODE and SDE versions with mixture Student-t distributions for observation dynamics.

## File Structure

```
graph_process/
├── config.py              # Parameter configuration system
├── data_loader.py         # Data loading with simulation support
├── training.py            # Training module with all loss functions
├── train.py              # Main training script
├── base_components.py    # Base components (discretization, networks)
├── dynamic_graph.py      # Dynamic causal graph (ODE/SDE)
├── observation_dynamics.py # Observation dynamics transformation
├── integrated_model.py   # Integrated model
├── utils/                # Utility modules
│   ├── __init__.py
│   ├── metrics.py        # Evaluation metrics (FDR, TPR, SHD, MSE, MDD)
│   └── simulation.py     # Dynamic LSEM simulation
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Quick Start

### Installation

#### 1. Install PyTorch with CUDA support
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Install other dependencies
```bash
pip install -r requirements.txt
```

#### 3. Verify CUDA installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Note**: If you see "CUDA available: False", you have installed the CPU-only version of PyTorch. See `CUDA_INSTALLATION_GUIDE.md` for detailed instructions.

### Training
```bash
# Default training
python train.py

# Debug mode (quick test)
python train.py --config_type debug

# Custom parameters
python train.py --d 5 --graph_type sde --epochs 100 --lr 0.001
```

## Features

### Complete Training Objectives
- **Data Likelihood Loss**: `L_data = -Σ log p(x_i(t) | X_≤t, G_≤t)`
- **Intervention Consistency Loss**: `L_int = -Σ log p(x_i(t) | do(x_I(t)), X_≤t, G_≤t)`
- **Counterfactual Consistency Loss**: `L_cf = D(p_cf, p_int)`
- **Graph Regularization**: Sparsity, temporal smoothness, acyclicity

### Training Features
- **Progress Bars**: Real-time training progress with tqdm
- **Weights & Biases**: Complete experiment tracking and monitoring
- **Multi-level Progress**: Epoch and batch-level progress tracking
- **Real-time Metrics**: Live loss and performance metrics
- **Checkpointing**: Model saving and loading
- **Early Stopping**: Prevent overfitting
- **Comprehensive Logging**: Detailed training logs

### Model Components
- **Dynamic Causal Graph**: ODE/SDE evolution of causal relations
- **Observation Dynamics**: Mixture Student-t distributions
- **Time Series Encoding**: RNN/GRU/LSTM encoders
- **Graph Aggregation**: Message passing with causal structure

### Data Generation
- **Dynamic LSEM**: Linear Structural Equation Model with time-varying causal relations
- **Graph Scenarios**: S1 (single edge) and S2 (Erdos-Renyi graphs)
- **Causal Functions**: F1 (cosine) and F2 (quadratic) time-varying functions
- **Interventions**: Explicit variable interventions
- **Counterfactuals**: Counterfactual scenario generation

### Evaluation Metrics
- **Graph Discovery**: FDR (False Discovery Rate), TPR (True Positive Rate), SHD (Structural Hamming Distance)
- **Prediction**: MSE (Mean Squared Error), MAE (Mean Absolute Error), MDD (Maximum Displacement Distance)
- **Temporal**: Metrics averaged over time sequences
- **Intervention**: Metrics specific to intervention scenarios

## Configuration

All parameters are configurable through the config system:

```python
from config import get_default_config, get_debug_config

# Use predefined configurations
config = get_debug_config()  # For quick testing
config = get_default_config()  # For standard training

# Or create custom configuration
config = Config()
config.model.d = 5
config.training.num_epochs = 100
```

## Usage Examples

### Basic Training
```python
from config import get_default_config
from integrated_model import IntegratedDynamicCausalModel
from data_loader import create_data_loaders
from training import Trainer

# Load configuration
config = get_default_config()

# Create model
model = IntegratedDynamicCausalModel(
    d=config.model.d,
    graph_type=config.model.graph_type,
    encoder_type=config.model.encoder_type
)

# Create data loaders
train_loader, val_loader, test_loader = create_data_loaders(
    config.data, config.model.d, device
)

# Train
trainer = Trainer(config, model)
trainer.train(train_loader, val_loader)
```

### Custom Configuration
```python
from config import Config

config = Config()
config.model.d = 4
config.model.graph_type = "sde"
config.model.encoder_type = "gru"
config.training.num_epochs = 50
config.training.learning_rate = 0.001
config.data.batch_size = 32
```

## Command Line Interface

```bash
# Training options
python train.py --config_type debug --epochs 10
python train.py --d 5 --graph_type sde --lr 0.001
python train.py --resume checkpoints/best_model.pt

# Dynamic LSEM simulation
python train.py --simulation_config S1_F1 --epochs 50
python train.py --scenario S2 --function_type F2 --epochs 100

# With Weights & Biases
python train.py --use_wandb --wandb_project "my-experiment" --epochs 50
python train.py --use_wandb --simulation_config S1_F1 --wandb_tags "S1" "F1" --epochs 100

# Test only
python train.py --test_only --resume checkpoints/best_model.pt
```

## Mathematical Formulation

The model implements the complete mathematical framework:

1. **Graph Evolution**: `dA(t) = f_θ_f(A(t), t) dt + g_θ_g(A(t), t) dW(t)`
2. **Graph Transformation**: `G(t) = σ(A(t)) ⊙ (1 - I)`
3. **Observation Dynamics**: `p(X_i(t+Δt)) = Σ_k π_{i,k}(t) T(·; μ_{i,k}(t), σ_{i,k}(t), ν_{i,k}(t))`
4. **Training Objective**: Combined likelihood and regularization terms

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib (optional, for visualization)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{dynamic_causal_graph_modeling,
  title={Dynamic Causal Graph Modeling},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/dynamic-causal-graph-modeling}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This implementation is provided for research purposes.
