"""
Dynamic Linear Structural Equation Model (Dynamic LSEM) Simulation

This module implements the Dynamic LSEM data generating process as described
in the paper, supporting different graph scenarios and causal relation functions.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import random
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """Configuration for Dynamic LSEM simulation."""
    p: int = 5  # Number of variables
    m: int = 30  # Number of observations per time point
    T: int = 10  # Number of time stamps
    n_realizations: int = 30  # Number of realizations
    
    # Graph scenarios
    scenario: str = "S1"  # "S1" or "S2"
    
    # Causal relation functions
    function_type: str = "F1"  # "F1" (cosine) or "F2" (quadratic)
    
    # Graph parameters for S2
    expected_degree: int = 4  # Expected degree for Erdos-Renyi model
    edge_prob: float = 0.3  # Edge probability for Erdos-Renyi model
    
    # Noise parameters
    noise_std: float = 0.1  # Standard deviation of Gaussian noise
    
    # Time parameters
    time_range: Tuple[float, float] = (0.0, 1.0)  # Time range [0, 1]


class DynamicLSEMSimulator:
    """Simulator for Dynamic Linear Structural Equation Model."""
    
    def __init__(self, config: SimulationConfig, device: torch.device = None):
        self.config = config
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Generate time points
        self.time_points = torch.linspace(
            config.time_range[0], config.time_range[1], config.T, device=self.device
        )
        
        # Generate true causal graph
        self.true_graph = self._generate_true_graph()
        
        # Generate causal relation functions
        self.causal_functions = self._generate_causal_functions()
    
    def _generate_true_graph(self) -> torch.Tensor:
        """Generate the true underlying causal graph."""
        p = self.config.p
        graph = torch.zeros(p, p, device=self.device)
        
        if self.config.scenario == "S1":
            # S1: Only one edge (A -> Y), where A=0, Y=1
            graph[0, 1] = 1.0  # A -> Y
            
        elif self.config.scenario == "S2":
            # S2: Erdos-Renyi model with expected degree
            # Generate random edges
            for i in range(p):
                for j in range(p):
                    if i != j:  # No self-loops
                        if random.random() < self.config.edge_prob:
                            graph[i, j] = 1.0
            
            # Ensure expected degree is approximately met
            current_edges = torch.sum(graph).item()
            target_edges = self.config.expected_degree * p / 2  # Undirected equivalent
            
            if current_edges < target_edges * 0.8:  # Too sparse
                # Add more edges randomly
                while torch.sum(graph).item() < target_edges * 0.9:
                    i, j = random.randint(0, p-1), random.randint(0, p-1)
                    if i != j and graph[i, j] == 0:
                        graph[i, j] = 1.0
            elif current_edges > target_edges * 1.2:  # Too dense
                # Remove some edges randomly
                while torch.sum(graph).item() > target_edges * 1.1:
                    edges = torch.nonzero(graph, as_tuple=True)
                    if len(edges[0]) > 0:
                        idx = random.randint(0, len(edges[0]) - 1)
                        graph[edges[0][idx], edges[1][idx]] = 0.0
        
        return graph
    
    def _generate_causal_functions(self) -> Dict[Tuple[int, int], callable]:
        """Generate causal relation functions for each edge."""
        functions = {}
        
        # Get all edges in the graph
        edges = torch.nonzero(self.true_graph, as_tuple=True)
        
        for i, j in zip(edges[0], edges[1]):
            i, j = i.item(), j.item()
            
            if self.config.function_type == "F1":
                # F1: Cosine function f1(t) = cos(4πt) * 0.8
                def cosine_func(t, i=i, j=j):
                    return torch.cos(4 * torch.pi * t) * 0.8
                functions[(i, j)] = cosine_func
                
            elif self.config.function_type == "F2":
                # F2: Quadratic function f2(t) = -10 + (20/(5-t))^2
                def quadratic_func(t, i=i, j=j):
                    # Avoid division by zero
                    t_safe = torch.clamp(t, min=0.01, max=4.99)
                    return -10 + (20 / (5 - t_safe)) ** 2
                functions[(i, j)] = quadratic_func
        
        return functions
    
    def _evaluate_causal_function(self, i: int, j: int, t: float) -> float:
        """Evaluate the causal function for edge (i,j) at time t."""
        if (i, j) in self.causal_functions:
            t_tensor = torch.tensor(t, device=self.device)
            return self.causal_functions[(i, j)](t_tensor).item()
        return 0.0
    
    def generate_single_realization(self) -> Dict[str, torch.Tensor]:
        """Generate a single realization of the Dynamic LSEM."""
        p = self.config.p
        m = self.config.m
        T = self.config.T
        
        # Initialize data tensor: (T, m, p)
        X = torch.zeros(T, m, p, device=self.device)
        
        # Generate data for each time point
        for t_idx, t in enumerate(self.time_points):
            t_val = t.item()
            
            # Generate data for each observation at this time point
            for obs in range(m):
                # Initialize with noise
                X[t_idx, obs, :] = torch.randn(p, device=self.device) * self.config.noise_std
                
                # Apply causal relations
                for i in range(p):
                    causal_effect = 0.0
                    
                    # Sum over all parents of variable i
                    for j in range(p):
                        if self.true_graph[j, i] > 0:  # j is a parent of i
                            # Get causal strength at time t
                            causal_strength = self._evaluate_causal_function(j, i, t_val)
                            
                            # Add causal effect from parent j
                            if t_idx > 0:  # Use previous time point
                                causal_effect += causal_strength * X[t_idx-1, obs, j]
                            else:  # First time point, use current value
                                causal_effect += causal_strength * X[t_idx, obs, j]
                    
                    # Update variable i with causal effect
                    X[t_idx, obs, i] += causal_effect
        
        return {
            'X': X,  # (T, m, p)
            'time_points': self.time_points,  # (T,)
            'true_graph': self.true_graph,  # (p, p)
            'causal_functions': self.causal_functions
        }
    
    def generate_all_realizations(self) -> List[Dict[str, torch.Tensor]]:
        """Generate all realizations of the Dynamic LSEM."""
        realizations = []
        
        for realization_idx in range(self.config.n_realizations):
            realization = self.generate_single_realization()
            realization['realization_idx'] = realization_idx
            realizations.append(realization)
        
        return realizations
    
    def generate_batch_data(self, batch_size: int = None) -> Dict[str, torch.Tensor]:
        """Generate batch data for training."""
        if batch_size is None:
            batch_size = self.config.n_realizations
        
        # Generate all realizations
        realizations = self.generate_all_realizations()
        
        # Stack data from multiple realizations
        X_batch = torch.stack([r['X'] for r in realizations[:batch_size]])  # (batch_size, T, m, p)
        time_points = realizations[0]['time_points']  # (T,)
        true_graph = realizations[0]['true_graph']  # (p, p)
        
        # Reshape for training: (batch_size, T*m, p)
        batch_size, T, m, p = X_batch.shape
        X_reshaped = X_batch.view(batch_size, T * m, p)
        
        # Create timestamps for each observation
        timestamps = time_points.unsqueeze(0).repeat_interleave(m, dim=0).flatten()  # (T*m,)
        timestamps_batch = timestamps.unsqueeze(0).repeat(batch_size, 1)  # (batch_size, T*m)
        
        return {
            'X': X_reshaped,  # (batch_size, T*m, p)
            'timestamps': timestamps_batch,  # (batch_size, T*m)
            'true_graph': true_graph,  # (p, p)
            'time_points': time_points,  # (T,)
            'config': self.config
        }
    
    def add_interventions(self, data: Dict[str, torch.Tensor], 
                         intervention_prob: float = 0.1) -> Dict[str, torch.Tensor]:
        """Add interventions to the generated data."""
        X = data['X']  # (batch_size, T*m, p)
        batch_size, seq_len, p = X.shape
        
        # Initialize intervention masks and values
        interventions = torch.zeros_like(X)
        intervention_values = torch.zeros_like(X)
        
        for b in range(batch_size):
            for t in range(seq_len):
                if random.random() < intervention_prob:
                    # Randomly select variables to intervene on
                    num_interventions = random.randint(1, min(3, p))
                    intervention_vars = random.sample(range(p), num_interventions)
                    
                    for var in intervention_vars:
                        interventions[b, t, var] = 1.0
                        # Set intervention value (e.g., set to 0 or random value)
                        intervention_values[b, t, var] = random.uniform(-2.0, 2.0)
        
        # Apply interventions
        X_intervened = X.clone()
        intervention_mask = interventions > 0
        X_intervened[intervention_mask] = intervention_values[intervention_mask]
        
        # Add to data dictionary
        data['X_original'] = X
        data['X'] = X_intervened
        data['interventions'] = interventions
        data['intervention_values'] = intervention_values
        
        return data
    
    def get_graph_evolution(self) -> torch.Tensor:
        """Get the evolution of causal strengths over time."""
        T = self.config.T
        p = self.config.p
        
        # Initialize graph evolution tensor
        graph_evolution = torch.zeros(T, p, p, device=self.device)
        
        # Get all edges
        edges = torch.nonzero(self.true_graph, as_tuple=True)
        
        for t_idx, t in enumerate(self.time_points):
            t_val = t.item()
            
            # Copy base graph structure
            graph_evolution[t_idx] = self.true_graph.clone()
            
            # Update edge weights with time-varying functions
            for i, j in zip(edges[0], edges[1]):
                i, j = i.item(), j.item()
                causal_strength = self._evaluate_causal_function(i, j, t_val)
                graph_evolution[t_idx, i, j] = causal_strength
        
        return graph_evolution
    
    def print_summary(self):
        """Print simulation summary."""
        print("Dynamic LSEM Simulation Summary")
        print("=" * 40)
        print(f"Scenario: {self.config.scenario}")
        print(f"Function Type: {self.config.function_type}")
        print(f"Variables (p): {self.config.p}")
        print(f"Observations per time (m): {self.config.m}")
        print(f"Time stamps (T): {self.config.T}")
        print(f"Realizations: {self.config.n_realizations}")
        print(f"True graph edges: {torch.sum(self.true_graph).item()}")
        print(f"Graph density: {torch.sum(self.true_graph).item() / (self.config.p * (self.config.p - 1)):.3f}")
        
        if self.config.scenario == "S2":
            print(f"Expected degree: {self.config.expected_degree}")
        
        print(f"Time range: {self.config.time_range}")
        print(f"Time points: {self.time_points.detach().cpu().numpy()}")


def create_simulation_configs(d: int = 4) -> Dict[str, SimulationConfig]:
    """Create predefined simulation configurations.
    
    Args:
        d: Number of variables (should match model's d parameter)
    """
    configs = {}
    
    # S1 + F1: Single edge with cosine function
    configs['S1_F1'] = SimulationConfig(
        scenario="S1",
        function_type="F1",
        p=d, m=30, T=10, n_realizations=30
    )
    
    # S1 + F2: Single edge with quadratic function
    configs['S1_F2'] = SimulationConfig(
        scenario="S1",
        function_type="F2",
        p=d, m=30, T=10, n_realizations=30
    )
    
    # S2 + F1: Erdos-Renyi graph with cosine functions
    configs['S2_F1'] = SimulationConfig(
        scenario="S2",
        function_type="F1",
        p=d, m=30, T=10, n_realizations=30,
        expected_degree=4, edge_prob=0.3
    )
    
    # S2 + F2: Erdos-Renyi graph with quadratic functions
    configs['S2_F2'] = SimulationConfig(
        scenario="S2",
        function_type="F2",
        p=d, m=30, T=10, n_realizations=30,
        expected_degree=4, edge_prob=0.3
    )
    
    return configs


# Example usage
if __name__ == "__main__":
    # Create simulator
    config = SimulationConfig(scenario="S1", function_type="F1")
    simulator = DynamicLSEMSimulator(config)
    
    # Print summary
    simulator.print_summary()
    
    # Generate data
    data = simulator.generate_batch_data(batch_size=5)
    print(f"\nGenerated data shape: {data['X'].shape}")
    print(f"Timestamps shape: {data['timestamps'].shape}")
    print(f"True graph:\n{data['true_graph'].detach().cpu().numpy()}")
    
    # Get graph evolution
    graph_evolution = simulator.get_graph_evolution()
    print(f"\nGraph evolution shape: {graph_evolution.shape}")
    print(f"Graph evolution at t=0:\n{graph_evolution[0].detach().cpu().numpy()}")
    print(f"Graph evolution at t=5:\n{graph_evolution[5].detach().cpu().numpy()}")
