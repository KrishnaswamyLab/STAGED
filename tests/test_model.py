import unittest
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.staged import STAGED


class TestSTAGEDModel(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for model tests"""
        # Test parameters
        self.num_genes = 5
        self.hidden_dim = 32
        self.num_gat_layers = 2
        self.num_mlp_layers = 2
        self.delta_gl = delta_lr = delta_rg = delta_gg = 1
        
        # Create a STAGED model
        self.model = STAGED(
            num_genes=self.num_genes,
            hidden_dim=self.hidden_dim,
            num_gat_layers=self.num_gat_layers,
            num_mlp_layers=self.num_mlp_layers,
            delta_gl=self.delta_gl,
            delta_lr=delta_lr,
            delta_rg=delta_rg,
            delta_gg=delta_gg
        )
        
        # Create sample cell graphs
        self.cell_graphs = self._create_sample_graphs(num_cells=2)
        
        # Create sample gene expression history
        self.gene_expression_history = {}
        for cell_idx in range(2):
            cell_id = f"cell_{cell_idx}"
            self.gene_expression_history[cell_id] = {}
            for gene_idx in range(self.num_genes):
                self.gene_expression_history[cell_id][gene_idx] = {}
                for t in range(5):
                    self.gene_expression_history[cell_id][gene_idx][t] = np.random.normal(0, 1)
        
        # Create sample cell positions
        self.cell_positions = {}
        for cell_idx in range(2):
            cell_id = f"cell_{cell_idx}"
            self.cell_positions[cell_id] = {}
            for t in range(5):
                self.cell_positions[cell_id][t] = [np.random.uniform(-10, 10), np.random.uniform(-10, 10)]
    
    def _create_sample_graphs(self, num_cells):
        """Helper to create sample PyTorch Geometric graphs"""
        cell_graphs = []
        
        for cell_idx in range(num_cells):
            # Create a graph with 20 nodes: 
            # - 5 gene nodes
            # - 5 receptor nodes
            # - 5 output ligand nodes
            # - 5 input ligand nodes
            num_nodes = 20
            
            # Edge indices (source, target)
            edge_index = torch.tensor([
                # Gene -> Ligand connections
                [0, 10], [1, 11], [2, 12], [3, 13], [4, 14],
                # Receptor -> Gene connections
                [5, 0], [5, 1], [5, 2], [5, 3], [5, 4],
                [6, 0], [6, 1], [6, 2], [6, 3], [6, 4],
                [7, 0], [7, 1], [7, 2], [7, 3], [7, 4],
                [8, 0], [8, 1], [8, 2], [8, 3], [8, 4],
                [9, 0], [9, 1], [9, 2], [9, 3], [9, 4],
                # Input Ligand -> Receptor connections
                [15, 5], [16, 6], [17, 7], [18, 8], [19, 9]
            ], dtype=torch.long)
            
            # Node features (random)
            x = torch.randn(num_nodes, 1)
            
            # Create a PyG Data object
            graph = Data(x=x, edge_index=edge_index.t().contiguous())
            
            # Store gene node indices
            graph.gene_node_indices = [0, 1, 2, 3, 4]
            
            cell_graphs.append(graph)
        
        return cell_graphs
    
    def test_initialization(self):
        """Test model initialization"""
        # Check attribute initialization
        self.assertEqual(self.model.num_genes, self.num_genes)
        self.assertEqual(self.model.hidden_dim, self.hidden_dim)
        
        # Check if GAT layers were created
        self.assertEqual(len(self.model.gat_layers), self.num_gat_layers)
        
        # Check if MLP layers were created
        self.assertEqual(len(self.model.mlp_layers), self.num_mlp_layers)
        
        # Check if first GAT layer has correct input dimension
        self.assertEqual(self.model.gat_layers[0].in_channels, self.model.input_dim)
        
        # Check if last MLP layer outputs a scalar
        self.assertEqual(self.model.mlp_layers[-1].out_features, 1)
    
    def test_forward_pass(self):
        """Test forward pass through the model"""
        # Run forward pass
        predictions, attention = self.model(self.cell_graphs, self.gene_expression_history, self.cell_positions)
        
        # Check predictions structure
        self.assertEqual(len(predictions), len(self.cell_graphs))
        
        # Check gene predictions for each cell
        for cell_idx in range(len(self.cell_graphs)):
            # Should have predictions for each gene
            for gene_idx in range(self.num_genes):
                self.assertIn(gene_idx, predictions[cell_idx])
                # Each prediction should be a scalar
                self.assertEqual(predictions[cell_idx][gene_idx].shape, torch.Size([1, 1]))
        
        # Check attention weights
        self.assertTrue(attention is not None)
    
    def test_get_t_init(self):
        """Test the get_t_init method"""
        # Set different time lags
        model = STAGED(
            num_genes=5,
            delta_gl=1,
            delta_lr=2,
            delta_rg=3,
            delta_gg=4
        )
        
        # t_init should be the maximum of all time lags
        self.assertEqual(model.get_t_init(), 4)


if __name__ == '__main__':
    unittest.main() 