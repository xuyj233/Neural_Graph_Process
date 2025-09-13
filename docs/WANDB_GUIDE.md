# Weights & Biases (wandb) Usage Guide

## 🎯 Feature Overview

Weights & Biases integration provides comprehensive experiment tracking and monitoring capabilities:

### 📊 Monitoring Metrics
- **Iteration-level monitoring**: Loss changes for each batch
- **Epoch-level monitoring**: Evaluation metrics for each epoch
- **Real-time visualization**: Loss curves, evaluation metric trends
- **Model architecture**: Parameter counts, model structure information

### 🔧 Configuration Options

#### Basic Configuration

```python
# In config.py
@dataclass
class ExperimentConfig:
    use_wandb: bool = True  # Whether to use wandb
    wandb_project: str = "dynamic-causal-graph"  # Project name
    wandb_entity: str = None  # Username or team name
    wandb_tags: List[str] = ["causal-graph", "time-series"]  # Tags
    wandb_notes: str = ""  # Experiment description
```

#### Command Line Configuration

```bash
# Enable wandb
python main.py --use_wandb

# Disable wandb
python main.py --no_wandb

# Custom project name
python main.py --wandb_project "my-experiment"

# Set team/username
python main.py --wandb_entity "my-team"

# Add tags
python main.py --wandb_tags "experiment1" "baseline"

# Add description
python main.py --wandb_notes "Testing new architecture"
```

## 📈 Monitoring Metrics Details

### Iteration-level Metrics (per batch)

```python
# Training losses
'train/iteration/total_loss': Total loss
'train/iteration/data_loss': Data likelihood loss
'train/iteration/intervention_loss': Intervention consistency loss
'train/iteration/counterfactual_loss': Counterfactual consistency loss
'train/iteration/sparsity_loss': Sparsity regularization loss
'train/iteration/smoothness_loss': Temporal smoothness loss
'train/iteration/acyclicity_loss': Acyclicity constraint loss

# Training status
'train/iteration/learning_rate': Learning rate
'train/iteration/global_step': Global step count
```

### Epoch-level Metrics (per epoch)

```python
# Training losses (epoch average)
'train/epoch/total_loss': Total loss
'train/epoch/data_loss': Data likelihood loss
'train/epoch/intervention_loss': Intervention consistency loss
'train/epoch/counterfactual_loss': Counterfactual consistency loss
'train/epoch/sparsity_loss': Sparsity regularization loss
'train/epoch/smoothness_loss': Temporal smoothness loss
'train/epoch/acyclicity_loss': Acyclicity constraint loss

# Validation losses (epoch average)
'val/epoch/total_loss': Total loss
'val/epoch/data_loss': Data likelihood loss
'val/epoch/intervention_loss': Intervention consistency loss
'val/epoch/counterfactual_loss': Counterfactual consistency loss
'val/epoch/sparsity_loss': Sparsity regularization loss
'val/epoch/smoothness_loss': Temporal smoothness loss
'val/epoch/acyclicity_loss': Acyclicity constraint loss

# Training status
'train/epoch/best_val_loss': Best validation loss
'train/epoch/learning_rate': Learning rate
'train/epoch/epoch_time': Epoch time
'train/epoch/patience_counter': Early stopping counter
```

### Evaluation Metrics (per epoch)

```python
# Prediction metrics
'eval/prediction/mse': Mean squared error
'eval/prediction/mae': Mean absolute error
'eval/prediction/mdd': Maximum displacement distance
'eval/prediction/correlation': Correlation coefficient

# Graph discovery metrics
'eval/graph/tpr': True positive rate
'eval/graph/fdr': False discovery rate
'eval/graph/shd': Structural Hamming distance
'eval/graph/precision': Precision
'eval/graph/f1': F1 score
```

### Model Information

```python
'model/total_parameters': Total parameter count
'model/trainable_parameters': Trainable parameter count
'model/parameter_ratio': Trainable parameter ratio
```

## 🚀 Usage Examples

### 1. Basic Usage

```bash
# Use default wandb configuration
python main.py --use_wandb --epochs 10
```

### 2. Custom Project

```bash
# Create new project
python main.py --use_wandb --wandb_project "my-causal-graph" --epochs 20
```

### 3. Team Collaboration

```bash
# Set team
python main.py --use_wandb --wandb_entity "my-team" --wandb_project "team-project"
```

### 4. Experiment Comparison

```bash
# Experiment 1: S1 scenario
python main.py --scenario S1 --use_wandb --wandb_tags "S1" "baseline"

# Experiment 2: S2 scenario
python main.py --scenario S2 --use_wandb --wandb_tags "S2" "baseline"
```

### 5. Debug Mode

```bash
# Debug experiment
python main.py --config_type debug --use_wandb --wandb_tags "debug" --epochs 1
```

## 📊 Visualization Features

### 1. Loss Curves
- Training and validation loss comparison
- Individual loss component trends
- Learning rate change curves

### 2. Evaluation Metrics
- Prediction performance metrics (MSE, MAE, MDD)
- Graph discovery performance metrics (TPR, FDR, SHD)
- Metric trends during training

### 3. Model Analysis
- Parameter count statistics
- Model complexity analysis
- Training efficiency metrics

### 4. Experiment Comparison
- Multi-experiment comparison charts
- Hyperparameter impact analysis
- Best experiment identification

## 🔧 Advanced Features

### 1. Hyperparameter Sweeps

```python
# Set up sweep in wandb interface
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val/epoch/total_loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'min': 1e-4, 'max': 1e-2},
        'batch_size': {'values': [16, 32, 64]},
        'hidden_dim': {'values': [32, 64, 128]}
    }
}
```

### 2. Model Saving

```python
# Automatically save best model to wandb
wandb.save('checkpoints/best_model.pt')
```

### 3. Experiment Reports

```python
# Generate experiment report
wandb.finish()
```

## 📋 Best Practices

### 1. Naming Conventions
- Use descriptive experiment names
- Include key hyperparameter information
- Use version numbers for identification

### 2. Tag Management
- Use consistent tag system
- Include experiment type, dataset, model version
- Facilitate filtering and comparison

### 3. Monitoring Frequency
- Iteration-level: Each batch (real-time monitoring)
- Epoch-level: Each epoch (evaluation metrics)
- Adjust monitoring frequency as needed

### 4. Resource Management
- Regularly clean up unnecessary experiments
- Use team accounts for permission management
- Set up reasonable project structure

## 🎉 Benefits

1. **Real-time Monitoring**: Training process visualization
2. **Experiment Management**: Systematic experiment recording
3. **Team Collaboration**: Shared experiment results
4. **Performance Analysis**: Detailed metric analysis
5. **Model Comparison**: Multi-experiment comparison features
6. **Reproducibility**: Complete experiment configuration recording

Now you can enjoy professional experiment tracking and monitoring experience!