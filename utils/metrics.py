"""
Evaluation Metrics Module for Dynamic Causal Graph Modeling

This module implements evaluation metrics for both causal graph discovery
and time series prediction performance.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import itertools


class MetricsCalculator:
    """Calculator for evaluation metrics."""
    
    def __init__(self, device: torch.device = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
    
    def calculate_graph_metrics(self, true_graph: torch.Tensor, pred_graph: torch.Tensor, 
                              threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate causal graph discovery metrics.
        
        Args:
            true_graph: True causal graph of shape (d, d)
            pred_graph: Predicted causal graph of shape (d, d)
            threshold: Threshold for binarizing predicted graph
            
        Returns:
            Dictionary containing FDR, TPR, SHD metrics
        """
        # Convert to numpy for easier computation
        if isinstance(true_graph, torch.Tensor):
            true_graph = true_graph.detach().cpu().numpy()
        if isinstance(pred_graph, torch.Tensor):
            pred_graph = pred_graph.detach().cpu().numpy()
        
        # Binarize predicted graph
        pred_binary = (pred_graph > threshold).astype(int)
        true_binary = (true_graph > 0).astype(int)
        
        # Remove self-loops
        d = true_graph.shape[0]
        pred_binary[np.diag_indices(d)] = 0
        true_binary[np.diag_indices(d)] = 0
        
        # Calculate confusion matrix elements
        tp = np.sum((pred_binary == 1) & (true_binary == 1))  # True positives
        fp = np.sum((pred_binary == 1) & (true_binary == 0))  # False positives
        fn = np.sum((pred_binary == 0) & (true_binary == 1))  # False negatives
        tn = np.sum((pred_binary == 0) & (true_binary == 0))  # True negatives
        
        # Calculate metrics
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
        fdr = fp / (tp + fp) if (tp + fp) > 0 else 0.0  # False Discovery Rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
        
        # Structural Hamming Distance (SHD)
        shd = fp + fn  # Number of edge differences
        
        return {
            'TPR': tpr,
            'FDR': fdr,
            'SHD': shd,
            'Precision': precision,
            'F1': f1,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        }
    
    def calculate_prediction_metrics(self, true_values: torch.Tensor, 
                                   pred_values: torch.Tensor) -> Dict[str, float]:
        """
        Calculate time series prediction metrics.
        
        Args:
            true_values: True values of shape (batch_size, d) or (seq_len, d)
            pred_values: Predicted values of shape (batch_size, d) or (seq_len, d)
            
        Returns:
            Dictionary containing MSE, MAE, MDD metrics
        """
        if isinstance(true_values, torch.Tensor):
            true_values = true_values.detach().cpu().numpy()
        if isinstance(pred_values, torch.Tensor):
            pred_values = pred_values.detach().cpu().numpy()
        
        # Ensure same shape
        if true_values.shape != pred_values.shape:
            raise ValueError(f"Shape mismatch: true {true_values.shape} vs pred {pred_values.shape}")
        
        # Calculate MSE
        mse = np.mean((true_values - pred_values) ** 2)
        
        # Calculate MAE
        mae = np.mean(np.abs(true_values - pred_values))
        
        # Calculate MDD (Maximum Displacement Distance)
        # For multivariate time series, calculate MDD for each variable and take mean
        if len(true_values.shape) == 2:
            # Shape: (batch_size, d) or (seq_len, d)
            mdd_per_var = []
            for i in range(true_values.shape[1]):
                true_var = true_values[:, i]
                pred_var = pred_values[:, i]
                
                # Calculate maximum displacement for this variable
                max_displacement = np.max(np.abs(true_var - pred_var))
                mdd_per_var.append(max_displacement)
            
            mdd = np.mean(mdd_per_var)
        else:
            # For univariate case
            mdd = np.max(np.abs(true_values - pred_values))
        
        # Calculate correlation
        if true_values.size > 1:
            # Flatten for correlation calculation
            true_flat = true_values.flatten()
            pred_flat = pred_values.flatten()
            correlation = np.corrcoef(true_flat, pred_flat)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        return {
            'MSE': mse,
            'MAE': mae,
            'MDD': mdd,
            'Correlation': correlation
        }
    
    def calculate_temporal_graph_metrics(self, true_graphs: torch.Tensor, 
                                       pred_graphs: torch.Tensor,
                                       threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate metrics for temporal graph sequences.
        
        Args:
            true_graphs: True graph sequence of shape (T, d, d)
            pred_graphs: Predicted graph sequence of shape (T, d, d)
            threshold: Threshold for binarizing predicted graphs
            
        Returns:
            Dictionary containing average metrics over time
        """
        if isinstance(true_graphs, torch.Tensor):
            true_graphs = true_graphs.detach().cpu().numpy()
        if isinstance(pred_graphs, torch.Tensor):
            pred_graphs = pred_graphs.detach().cpu().numpy()
        
        T = true_graphs.shape[0]
        metrics_over_time = {
            'TPR': [],
            'FDR': [],
            'SHD': [],
            'Precision': [],
            'F1': []
        }
        
        # Calculate metrics for each time step
        for t in range(T):
            metrics_t = self.calculate_graph_metrics(
                true_graphs[t], pred_graphs[t], threshold
            )
            for key in metrics_over_time:
                metrics_over_time[key].append(metrics_t[key])
        
        # Calculate average metrics
        avg_metrics = {}
        for key, values in metrics_over_time.items():
            avg_metrics[f'Avg_{key}'] = np.mean(values)
            avg_metrics[f'Std_{key}'] = np.std(values)
        
        return avg_metrics
    
    def calculate_intervention_metrics(self, true_values: torch.Tensor,
                                     pred_values: torch.Tensor,
                                     intervention_mask: torch.Tensor) -> Dict[str, float]:
        """
        Calculate metrics specifically for intervention scenarios.
        
        Args:
            true_values: True values under intervention
            pred_values: Predicted values under intervention
            intervention_mask: Binary mask indicating intervened variables
            
        Returns:
            Dictionary containing intervention-specific metrics
        """
        if isinstance(intervention_mask, torch.Tensor):
            intervention_mask = intervention_mask.detach().cpu().numpy()
        
        # Calculate overall metrics
        overall_metrics = self.calculate_prediction_metrics(true_values, pred_values)
        
        # Calculate metrics for intervened variables only
        if intervention_mask.any():
            intervened_indices = np.where(intervention_mask > 0)
            
            if isinstance(true_values, torch.Tensor):
                true_intervened = true_values[intervened_indices]
                pred_intervened = pred_values[intervened_indices]
            else:
                true_intervened = true_values[intervened_indices]
                pred_intervened = pred_values[intervened_indices]
            
            intervention_metrics = self.calculate_prediction_metrics(
                true_intervened, pred_intervened
            )
            
            # Add prefix to distinguish intervention metrics
            for key, value in intervention_metrics.items():
                overall_metrics[f'Intervention_{key}'] = value
        
        return overall_metrics
    
    def calculate_comprehensive_metrics(self, true_graphs: torch.Tensor,
                                      pred_graphs: torch.Tensor,
                                      true_values: torch.Tensor,
                                      pred_values: torch.Tensor,
                                      intervention_mask: Optional[torch.Tensor] = None,
                                      threshold: float = 0.5) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            true_graphs: True graph sequence of shape (T, d, d)
            pred_graphs: Predicted graph sequence of shape (T, d, d)
            true_values: True time series values
            pred_values: Predicted time series values
            intervention_mask: Optional intervention mask
            threshold: Threshold for graph binarization
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Graph discovery metrics
        if true_graphs is not None and pred_graphs is not None:
            if len(true_graphs.shape) == 3:  # Temporal sequence
                graph_metrics = self.calculate_temporal_graph_metrics(
                    true_graphs, pred_graphs, threshold
                )
            else:  # Single graph
                graph_metrics = self.calculate_graph_metrics(
                    true_graphs, pred_graphs, threshold
                )
            metrics.update(graph_metrics)
        
        # Prediction metrics
        if true_values is not None and pred_values is not None:
            if intervention_mask is not None:
                pred_metrics = self.calculate_intervention_metrics(
                    true_values, pred_values, intervention_mask
                )
            else:
                pred_metrics = self.calculate_prediction_metrics(
                    true_values, pred_values
                )
            metrics.update(pred_metrics)
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float], title: str = "Evaluation Metrics"):
        """Print metrics in a formatted way."""
        print(f"\n{title}")
        print("=" * len(title))
        
        # Group metrics by category
        graph_metrics = {k: v for k, v in metrics.items() 
                        if k in ['TPR', 'FDR', 'SHD', 'Precision', 'F1', 'Avg_TPR', 'Avg_FDR', 'Avg_SHD']}
        pred_metrics = {k: v for k, v in metrics.items() 
                       if k in ['MSE', 'MAE', 'MDD', 'Correlation']}
        
        if graph_metrics:
            print("\nGraph Discovery Metrics:")
            for key, value in graph_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        if pred_metrics:
            print("\nPrediction Metrics:")
            for key, value in pred_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print()


def calculate_batch_metrics(metrics_calculator: MetricsCalculator,
                          true_graphs: List[torch.Tensor],
                          pred_graphs: List[torch.Tensor],
                          true_values: List[torch.Tensor],
                          pred_values: List[torch.Tensor],
                          threshold: float = 0.5) -> Dict[str, float]:
    """
    Calculate metrics for a batch of samples.
    
    Args:
        metrics_calculator: MetricsCalculator instance
        true_graphs: List of true graphs
        pred_graphs: List of predicted graphs
        true_values: List of true values
        pred_values: List of predicted values
        threshold: Threshold for graph binarization
        
    Returns:
        Dictionary containing average metrics across the batch
    """
    batch_metrics = {
        'TPR': [], 'FDR': [], 'SHD': [], 'Precision': [], 'F1': [],
        'MSE': [], 'MAE': [], 'MDD': [], 'Correlation': []
    }
    
    for i in range(len(true_graphs)):
        # Graph metrics
        graph_metrics = metrics_calculator.calculate_graph_metrics(
            true_graphs[i], pred_graphs[i], threshold
        )
        for key in ['TPR', 'FDR', 'SHD', 'Precision', 'F1']:
            batch_metrics[key].append(graph_metrics[key])
        
        # Prediction metrics
        pred_metrics = metrics_calculator.calculate_prediction_metrics(
            true_values[i], pred_values[i]
        )
        for key in ['MSE', 'MAE', 'MDD', 'Correlation']:
            batch_metrics[key].append(pred_metrics[key])
    
    # Calculate averages
    avg_metrics = {}
    for key, values in batch_metrics.items():
        avg_metrics[f'Avg_{key}'] = np.mean(values)
        avg_metrics[f'Std_{key}'] = np.std(values)
    
    return avg_metrics
