import unittest
import os
import sys
import torch
import networkx as nx
import numpy as np

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.graph_constructor import GraphConstructor
from models.staged import STAGED
from torch_geometric.data import Data


class TestFixes(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Sample data for graph constructor tests
        self.genes = [f"gene_{i}" for i in range(3)]
        self.lr_pairs = [("gene_0", "gene_1")]
        self.cell_types = {"cell_0": "type_A"}
        
        # Create prior GRN
        self.prior_grn = {"type_A": nx.DiGraph()}
        for gene in self.genes:
            self.prior_grn["type_A"].add_node(gene)
        
        # Sample gene expression history
        self.gene_expression_history = {
            "cell_0": {
                0: {0: 0.1, 1: 0.2},
                1: {0: 0.3, 1: 0.4},
                2: {0: 0.5, 1: 0.6}
            }
        }
        
        # Sample cell positions
        self.cell_positions = {
            "cell_0": {0: [0.0, 0.0], 1: [1.0, 1.0]}
        }
        
    def test_graph_constructor_node_index_fix(self):
        """Test that the fix for the NodeView.index issue works correctly"""
        # Create a graph constructor
        graph_constructor = GraphConstructor(
            genes=self.genes,
            ligand_receptor_pairs=self.lr_pairs,
            cell_type_assignments=self.cell_types,
            prior_grns=self.prior_grn
        )
        
        # Create a base graph
        base_graph = graph_constructor.construct_base_graph("cell_0")
        
        # Test that gene nodes are in the graph
        for gene in self.genes:
            self.assertIn(gene, base_graph.nodes())
        
        # Assign node features - this is where the original error occurred
        try:
            pyg_graph = graph_constructor.assign_node_features(
                graph=base_graph,
                cell_id="cell_0",
                time_point=1,
                gene_expression_history=self.gene_expression_history,
                delta_gl=1,
                delta_lr=1,
                delta_rg=1,
                delta_gg=1
            )
            # If we get here without an error, the fix worked
            self.assertTrue(True)
        except AttributeError as e:
            self.fail(f"assign_node_features raised AttributeError: {e}")
        
        # Check that gene_node_indices is correctly populated
        self.assertTrue(hasattr(pyg_graph, 'gene_node_indices'))
        self.assertEqual(len(pyg_graph.gene_node_indices), len(self.genes))
    
    def test_model_prediction_shape_fix(self):
        """Test that the fix for the model prediction shape works correctly"""
        # Create a model
        model = STAGED(
            num_genes=3,
            hidden_dim=16,
            num_gat_layers=1,
            num_mlp_layers=1
        )
        
        # Create a test graph
        edge_index = torch.tensor([
            [0, 1], [1, 2], [2, 0]  # Simple cycle
        ], dtype=torch.long).t().contiguous()
        
        x = torch.randn(3, 1)  # 3 nodes, 1 feature each
        
        graph = Data(x=x, edge_index=edge_index)
        graph.gene_node_indices = [0, 1, 2]  # All nodes are gene nodes
        
        # Run forward pass
        predictions, _ = model([graph], self.gene_expression_history, self.cell_positions)
        
        # Check that predictions have the correct shape
        for cell_idx in predictions:
            for gene_idx in predictions[cell_idx]:
                # Each prediction should have shape [1, 1] (not [1])
                self.assertEqual(predictions[cell_idx][gene_idx].shape, torch.Size([1, 1]))
                # Also check the type
                self.assertIsInstance(predictions[cell_idx][gene_idx], torch.Tensor)


if __name__ == '__main__':
    unittest.main() 