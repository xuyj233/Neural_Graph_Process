"""
Main Entry Point for Dynamic Causal Graph Modeling

This is the main entry point for the Dynamic Causal Graph Modeling system.
It provides a simple interface to run experiments and training.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.train import main

if __name__ == "__main__":
    import sys
    main()
