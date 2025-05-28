import unittest
import os
import sys
import torch
import numpy as np
from typing import Dict
import networkx as nx


# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.training import train_staged_model, TrainingConfig, ModelConfig
from test_graph_constructor import create_square_grid_data

# NOTE: This file tests next-step prediction training.
# For Neural ODE training tests, see test_training_ode.py

def create_simple_test_data(
    n_time_points: int = 15,
    n_cells: int = 4,
    n_genes: int = 6,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """
    Create a simple test dataset with predictable patterns.
    Each gene follows a simple sinusoidal pattern over time.
    """
    # Create time points
    time = torch.linspace(0, 2*np.pi, n_time_points).unsqueeze(-1).unsqueeze(-1)
    
    # Create gene expression patterns (sinusoidal with different frequencies)
    gene_patterns = torch.zeros((n_time_points, n_cells, n_genes), device=device)
    for g in range(n_genes):
        # Each gene has a different frequency
        freq = g + 1
        gene_patterns[..., g] = torch.sin(freq * time).squeeze(-1)
    
    # Create cell positions in a 2x2 grid
    cell_positions = torch.zeros((n_time_points, n_cells, 2), device=device)
    cell_positions[:, 0] = torch.tensor([0.0, 0.0])    # Cell 0: top-left
    cell_positions[:, 1] = torch.tensor([1.0, 0.0])    # Cell 1: top-right
    cell_positions[:, 2] = torch.tensor([0.0, 1.0])    # Cell 2: bottom-left
    cell_positions[:, 3] = torch.tensor([1.0, 1.0])    # Cell 3: bottom-right
    
    return {
        'gene_expression': gene_patterns,
        'cell_positions': cell_positions,
        'n_cells': n_cells
    }

class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and parameters once for all test methods."""
        print("\nSetting up test data and parameters...")
        
        # Set device
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {cls.device}")
        
        # Create simple test data
        cls.simple_data = create_simple_test_data(device=cls.device)
        
        # Create more complex test data using existing function
        cls.complex_data = create_square_grid_data()
        
        # Convert complex data tensors to proper device
        for key, value in cls.complex_data.items():
            if isinstance(value, torch.Tensor):
                cls.complex_data[key] = value.to(cls.device)
        
        # Model configuration
        cls.model_config = ModelConfig(
            hidden_dim=32,  # Smaller for testing
            num_gat_layers=1,
            num_mlp_layers=2,
            dropout=0.1
        )
        
        # Training configuration
        cls.config = TrainingConfig(
            max_iterations=10,  # Small number for testing
            learning_rate=0.01,
            batch_size=2,
            device=cls.device,
            model_config=cls.model_config
        )

    def test_simple_one_step_training(self):
        """Test training with simple sinusoidal data in one-step mode."""
        print("\nTesting one-step training with simple data...")
        
        # Get genes and pairs from simple data
        n_genes = self.simple_data['gene_expression'].shape[-1]
        genes = [f"gene_{i}" for i in range(n_genes)]
        
        # Create simple L-R pairs (first half are ligands, second half are receptors)
        mid = n_genes // 2
        ligand_receptor_pairs = [
            (f"gene_{i}", f"gene_{i+mid}")
            for i in range(mid)
        ]
        
        # Create simple R-G pairs
        receptor_gene_pairs = [
            (f"gene_{i+mid}", f"gene_{j}")
            for i in range(mid)
            for j in range(n_genes)
            if j != i+mid
        ]
        
        # Create simple cell type assignments and GRNs
        cell_type_assignments = torch.zeros(
            self.simple_data['n_cells'],
            dtype=torch.long,
            device=self.device
        )
        
        # Create a simple GRN where each gene regulates the next
        grn = nx.DiGraph()
        for i in range(n_genes):
            grn.add_node(f"gene_{i}")
            if i > 0:
                grn.add_edge(f"gene_{i-1}", f"gene_{i}")
        prior_grns = {0: grn}
        
        # Train model
        output = train_staged_model(
            data=self.simple_data,
            genes=genes,
            ligand_receptor_pairs=ligand_receptor_pairs,
            receptor_gene_pairs=receptor_gene_pairs,
            cell_type_assignments=cell_type_assignments,
            prior_grns=prior_grns,
            prediction_mode="one_step",
            config=self.config
        )
        
        # Check that training produced loss history
        self.assertEqual(len(output.loss_history), self.config.max_iterations)
        
        # Check that loss decreased
        self.assertLess(output.loss_history[-1], output.loss_history[0])
        print(f"Initial loss: {output.loss_history[0]:.6f}")
        print(f"Final loss: {output.loss_history[-1]:.6f}")

    def test_complex_one_step_training(self):
        """Test training with complex grid data in one-step mode."""
        print("\nTesting one-step training with complex data...")
        
        # Train model using complex data
        output = train_staged_model(
            data=self.complex_data,
            genes=self.complex_data['genes'],
            ligand_receptor_pairs=self.complex_data['ligand_receptor_pairs'],
            receptor_gene_pairs=self.complex_data['receptor_gene_pairs'],
            cell_type_assignments=self.complex_data['cell_type_assignments'],
            prior_grns=self.complex_data['prior_grns'],
            prediction_mode="one_step",
            config=self.config
        )
        
        # Check that training produced loss history
        self.assertEqual(len(output.loss_history), self.config.max_iterations)
        
        # Check that loss decreased
        self.assertLess(output.loss_history[-1], output.loss_history[0])
        print(f"Initial loss: {output.loss_history[0]:.6f}")
        print(f"Final loss: {output.loss_history[-1]:.6f}")

    def test_k_step_validation(self):
        """Test that k-step mode properly validates parameters."""
        print("\nTesting k-step parameter validation...")
        
        with self.assertRaises(ValueError):
            # Should raise error when k_steps not provided
            train_staged_model(
                data=self.simple_data,
                genes=[f"gene_{i}" for i in range(6)],
                ligand_receptor_pairs=[("gene_0", "gene_3")],
                receptor_gene_pairs=[("gene_3", "gene_1")],
                cell_type_assignments=torch.zeros(4),
                prior_grns={0: nx.DiGraph()},
                prediction_mode="k_step",
                config=self.config
            )
        
        with self.assertRaises(ValueError):
            # Should raise error when k_steps too large
            train_staged_model(
                data=self.simple_data,
                genes=[f"gene_{i}" for i in range(6)],
                ligand_receptor_pairs=[("gene_0", "gene_3")],
                receptor_gene_pairs=[("gene_3", "gene_1")],
                cell_type_assignments=torch.zeros(4),
                prior_grns={0: nx.DiGraph()},
                prediction_mode="k_step",
                k_steps=100,  # Larger than available steps
                config=self.config
            )

    def test_invalid_prediction_mode(self):
        """Test that invalid prediction mode raises error."""
        print("\nTesting invalid prediction mode handling...")
        
        with self.assertRaises(NotImplementedError):
            train_staged_model(
                data=self.simple_data,
                genes=[f"gene_{i}" for i in range(6)],
                ligand_receptor_pairs=[("gene_0", "gene_3")],
                receptor_gene_pairs=[("gene_3", "gene_1")],
                cell_type_assignments=torch.zeros(4),
                prior_grns={0: nx.DiGraph()},
                prediction_mode="invalid_mode",
                config=self.config
            )

if __name__ == '__main__':
    unittest.main(verbosity=2) 