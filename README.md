# Dynamic Causal Graph Modeling

A complete implementation for dynamic causal graph modeling, supporting both ODE and SDE versions of graph evolution, and observation dynamics transformation based on mixture Student-t distributions.

## Project Structure

```
graph_process/
├── main.py                 # Main entry point
├── config.py              # Configuration management
├── requirements.txt       # Dependencies
├── models/                # Model components
│   ├── __init__.py
│   ├── base_components.py      # Base components (discretization functions, neural networks)
│   ├── dynamic_graph.py        # Dynamic graph models (ODE/SDE)
│   ├── observation_dynamics.py # Observation dynamics transformation
│   └── integrated_model.py     # Integrated model
├── data/                  # Data processing
│   ├── __init__.py
│   └── data_loader.py          # Data loader
├── training/              # Training module
│   ├── __init__.py
│   └── trainer.py              # Trainer
├── experiments/           # Experiment scripts
│   ├── __init__.py
│   └── train.py                # Training script
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics
│   └── simulation.py           # Data simulation
├── docs/                  # Documentation
│   ├── CUDA_INSTALLATION_GUIDE.md
│   ├── WANDB_GUIDE.md
│   └── README.md
├── tests/                 # Test files
├── checkpoints/           # Model checkpoints
├── logs/                  # Log files
└── wandb/                 # Weights & Biases experiment tracking
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

#### Basic Training
```bash
python main.py --config_type debug --epochs 5
```

#### With Weights & Biases Tracking
```bash
python main.py --use_wandb --wandb_project "dynamic-causal-graph" --epochs 10
```

#### Specify Scenario and Function Type
```bash
python main.py --scenario S1 --function_type F1 --epochs 20
```

### 3. Configuration Options

- `--config_type`: Configuration type (debug/small/default/large)
- `--epochs`: Number of training epochs
- `--scenario`: Simulation scenario (S1/S2)
- `--function_type`: Causal function type (F1/F2)
- `--use_wandb`: Enable Weights & Biases tracking
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity name

## Key Features

### 1. Dynamic Causal Graph Modeling
- **ODE Version**: Deterministic graph evolution
- **SDE Version**: Stochastic graph evolution with noise and random perturbations

### 2. Observation Dynamics Transformation
- Time series encoders (RNN/GRU/LSTM)
- Graph-structured message aggregation
- Mixture Student-t distribution modeling

### 3. Training Objectives
- Data likelihood loss
- Intervention consistency loss
- Counterfactual consistency loss
- Graph regularization (sparsity, temporal smoothness, acyclicity)

### 4. Data Generation
- Dynamic LSEM simulator
- Support for interventions and counterfactual data
- Multiple scenarios and function types

### 5. Evaluation Metrics
- Graph discovery metrics: FDR, TPR, SHD, Precision, F1
- Prediction metrics: MSE, MAE, MDD, Correlation

### 6. Experiment Tracking
- Weights & Biases integration
- Real-time loss and metrics monitoring
- Model architecture logging

## Usage Examples

### Basic Training
```python
from experiments.train import main

# Run debug configuration
main(['--config_type', 'debug', '--epochs', '5'])
```

### Custom Configuration
```python
from config import get_default_config
from models import IntegratedDynamicCausalModel
from training import Trainer

# Create configuration
config = get_default_config()
config.model.d = 5
config.training.epochs = 10

# Create model
model = IntegratedDynamicCausalModel(config.model)

# Train
trainer = Trainer(model, config.training, device='cuda')
trainer.train(train_loader, val_loader)
```

## System Requirements

- Python 3.8+
- PyTorch 1.12+ (with CUDA support)
- CUDA 11.6+ (optional, for GPU acceleration)

## Troubleshooting

### GPU Issues
If you encounter GPU-related issues, please refer to `docs/CUDA_INSTALLATION_GUIDE.md`

### Weights & Biases Setup
For W&B usage, please refer to `docs/WANDB_GUIDE.md`

## License

MIT License