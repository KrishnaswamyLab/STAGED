import unittest
import torch
from models.staged import STAGED
from utils.graph_constructor import GraphConstructor
from utils.graph_data_handler import GraphDataHandler
from .test_graph_constructor import create_test_data

class TestGraphDataHandler(unittest.TestCase):
    def setUp(self):
        # Create test data
        self.data = create_test_data()
        
        # Initialize graph constructor
        self.graph_constructor = GraphConstructor(
            genes=self.data['genes'],
            ligand_receptor_pairs=self.data['ligand_receptor_pairs'],
            receptor_gene_pairs=self.data['receptor_gene_pairs'],
            cell_type_assignments=self.data['cell_type_assignments'],
            prior_grns=self.data['prior_grns']
        )
        
        # Define time lags
        self.delta_gl = 1
        self.delta_lr = 5
        self.delta_rg = 3
        self.delta_gg = 7
        
        # Set time point
        self.time_point = max(self.delta_gl, self.delta_lr, self.delta_rg, self.delta_gg)
        
        # Create model
        self.model = STAGED(
            num_genes=len(self.data['genes']),
            hidden_dim=64,
            num_gat_layers=1,
            num_mlp_layers=3,
            dropout=0.1,
            delta_gl=self.delta_gl,
            delta_lr=self.delta_lr,
            delta_rg=self.delta_rg,
            delta_gg=self.delta_gg,
            add_self_loops=False,
        )
        
        # Create graph data handler
        self.graph_handler = GraphDataHandler(self.model)
        
        # Prepare cell graphs
        self.cell_graphs = []
        for cell_idx in range(self.data['n_cells']):
            # Construct and update graph for each cell
            base_graph = self.graph_constructor.construct_base_graph(cell_idx)
            updated_graph = self.graph_constructor.update_graph_with_neighbors(
                base_graph, cell_idx, self.data['cell_positions'], 
                self.time_point, distance_threshold=10.0
            )
            # Assign features
            pyg_graph = self.graph_constructor.assign_node_features(
                updated_graph, cell_idx, self.time_point, 
                self.data['gene_expression'],
                self.delta_gl, self.delta_lr, self.delta_rg, self.delta_gg
            )
            self.cell_graphs.append(pyg_graph)

    # Add helper method to compare attention weights
    def compare_attention_weights(self, attn1, attn2, batch_size1, batch_size2):
        """
        Compare two sets of attention weights
        
        Args:
            attn1, attn2: Tuples of (edge_index, attention_values)
            batch_size1, batch_size2: Batch sizes used to generate the attention weights
        
        Returns:
            bool: True if attention weights are consistent
        """
        edge_index1, values1 = attn1
        edge_index2, values2 = attn2
        
        # Compare edge indices and attention values
        edges_match = torch.allclose(edge_index1, edge_index2)
        values_match = torch.allclose(values1, values2, rtol=1e-5, atol=1e-5)
        
        # Print basic statistics
        print(f"\nAttention Comparison (batch_size={batch_size1} vs batch_size={batch_size2}):")
        print(f"Edge indices shapes: {edge_index1.shape} vs {edge_index2.shape}")
        print(f"Attention values shapes: {values1.shape} vs {values2.shape}")
        
        if not edges_match:
            print("\nEdge indices mismatch:")
            print(f"Number of different edges: {(edge_index1 != edge_index2).sum() // 2}")
            # Print first few differences
            diff_mask = (edge_index1 != edge_index2).any(dim=0)
            diff_indices = torch.where(diff_mask)[0]
            if len(diff_indices) > 0:
                print("\nFirst few edge index differences:")
                for i in range(min(5, len(diff_indices))):
                    idx = diff_indices[i]
                    print(f"Edge {i}: {edge_index1[:,idx].tolist()} vs {edge_index2[:,idx].tolist()}")
        else:
            print("Edge indices match")
        
        if not values_match:
            abs_diff = torch.abs(values1 - values2)
            print("\nAttention values mismatch:")
            print(f"Max absolute difference: {torch.max(abs_diff)}")
            print(f"Mean absolute difference: {torch.mean(abs_diff)}")
            print(f"Median absolute difference: {torch.median(abs_diff)}")
            print(f"Standard deviation of difference: {torch.std(abs_diff)}")
            
            # Print histogram of differences
            diff_hist = torch.histogram(abs_diff, bins=10)
            print("\nHistogram of absolute differences:")
            for bin_idx in range(len(diff_hist.hist)):
                bin_count = diff_hist.hist[bin_idx]
                bin_edge = diff_hist.bin_edges[bin_idx]
                print(f"Bin {bin_idx} ({bin_edge:.2e}): {bin_count}")
            
            # Print first few largest differences
            largest_diff_indices = torch.argsort(abs_diff, descending=True)
            print("\nLargest differences:")
            for i in range(min(5, len(largest_diff_indices))):
                idx = largest_diff_indices[i]
                print(f"Edge {idx}: {values1[idx]:.6f} vs {values2[idx]:.6f} "
                      f"(diff: {abs_diff[idx]:.6f})")
                print(f"  Source->Target: {edge_index1[:,idx].tolist()}")
        else:
            print("Attention values match")
        
        return edges_match and values_match

    def test_batch_consistency(self):
        """Test if different batch sizes produce consistent results"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Ensure model is in eval mode to disable dropout
        self.model.eval()
        
        # Process with different batch sizes
        batch_sizes = [None, 1, 2, len(self.cell_graphs)]
        results = {}
        attention_results = {}
        
        for batch_size in batch_sizes:
            predictions, attention_weights, node_ptr = self.graph_handler.process_cell_graphs(
                self.cell_graphs,
                num_genes=len(self.data['genes']),
                batch_size=batch_size
            )
            results[batch_size] = predictions.detach()
            attention_results[batch_size] = attention_weights
        
        # Compare results between different batch sizes
        reference_pred = results[None]  # Use no batching as reference
        reference_attn = attention_results[None]  # Reference attention weights
        
        for batch_size, predictions in results.items():
            if batch_size is not None:
                # Compare predictions
                max_diff = torch.max(torch.abs(reference_pred - predictions))
                print(f"\nBatch size {batch_size} comparison:")
                print(f"Max prediction difference: {max_diff}")
                print(f"Reference shape: {reference_pred.shape}")
                print(f"Prediction shape: {predictions.shape}")
                
                if not torch.allclose(reference_pred, predictions, rtol=1e-5, atol=1e-5):
                    # Print detailed comparison for first mismatch
                    mismatch_mask = ~torch.isclose(reference_pred, predictions, rtol=1e-5, atol=1e-5)
                    mismatch_idx = torch.where(mismatch_mask)
                    if len(mismatch_idx[0]) > 0:
                        cell_idx, gene_idx = mismatch_idx[0][0], mismatch_idx[1][0]
                        print(f"\nFirst prediction mismatch at cell {cell_idx}, gene {gene_idx}:")
                        print(f"Reference value: {reference_pred[cell_idx, gene_idx]}")
                        print(f"Prediction value: {predictions[cell_idx, gene_idx]}")
                
                # Compare attention weights
                attention_match = self.compare_attention_weights(
                    reference_attn,
                    attention_results[batch_size],
                    None,
                    batch_size
                )
                
                # Assert predictions match
                self.assertTrue(
                    torch.allclose(reference_pred, predictions, rtol=1e-5, atol=1e-5),
                    f"Predictions with batch_size={batch_size} don't match reference"
                )
                
                # Assert attention weights match
                self.assertTrue(
                    attention_match,
                    f"Attention weights with batch_size={batch_size} don't match reference"
                )
                
                # Check shapes
                self.assertEqual(
                    predictions.shape,
                    (self.data['n_cells'], len(self.data['genes'])),
                    f"Wrong prediction shape for batch_size={batch_size}"
                )

    def test_device_handling(self):
        """Test if device handling works correctly"""
        if torch.cuda.is_available():
            # Create handler with explicit CPU device
            cpu_handler = GraphDataHandler(self.model, device=torch.device('cpu'))
            cpu_pred, _, _ = cpu_handler.process_cell_graphs(
                self.cell_graphs,
                num_genes=len(self.data['genes'])
            )
            
            # Create handler with CUDA device
            cuda_handler = GraphDataHandler(self.model, device=torch.device('cuda'))
            cuda_pred, _, _ = cuda_handler.process_cell_graphs(
                self.cell_graphs,
                num_genes=len(self.data['genes'])
            )
            
            # Compare results
            self.assertTrue(
                torch.allclose(cpu_pred, cuda_pred.cpu(), rtol=1e-5, atol=1e-5),
                "Predictions differ between CPU and CUDA"
            )

    def test_attention_weights(self):
        """Test if attention weights are properly returned"""
        predictions, attention_weights, node_ptr = self.graph_handler.process_cell_graphs(
            self.cell_graphs,
            num_genes=len(self.data['genes'])
        )
        
        # Check if attention weights are returned
        self.assertIsNotNone(attention_weights)
        
        # Check attention weights format
        # Attention should be a tuple of (edge_index, attention_values)
        self.assertTrue(isinstance(attention_weights, tuple))
        self.assertEqual(len(attention_weights), 2)
        
        edge_index, attn_values = attention_weights
        # Check if edge_index is a 2D tensor
        self.assertEqual(len(edge_index.shape), 2)
        # Check if attention values match number of edges
        self.assertEqual(attn_values.shape[0], edge_index.shape[1])
        # Check if attention values are single-channel
        self.assertEqual(attn_values.shape[1], 1)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        # Test with empty graph list
        with self.assertRaises(ValueError):
            self.graph_handler.process_cell_graphs([], num_genes=len(self.data['genes']))
        
        # Test with invalid batch size
        with self.assertRaises(ValueError):
            self.graph_handler.process_cell_graphs(
                self.cell_graphs,
                num_genes=len(self.data['genes']),
                batch_size=0
            )

    def test_attention_splitting(self):
        """Test if attention weights can be correctly split back into individual graphs"""
        # Get concatenated attention
        predictions, attention_weights, node_ptr = self.graph_handler.process_cell_graphs(
            self.cell_graphs,
            num_genes=len(self.data['genes'])
        )
        
        # Split attention weights
        split_attention = self.graph_handler.split_attention_by_graphs(attention_weights, node_ptr)
        
        # Check if we got the right number of graphs
        self.assertEqual(len(split_attention), len(self.cell_graphs))
        
        # Check each graph's attention
        for i, (graph_edges, graph_attn) in enumerate(split_attention):
            # Check that indices are within bounds for this graph
            self.assertTrue((graph_edges < self.cell_graphs[i].num_nodes).all())
            self.assertTrue((graph_edges >= 0).all())
            
            # Check shapes
            self.assertEqual(len(graph_edges.shape), 2)  # Should be 2D
            self.assertEqual(graph_edges.shape[0], 2)    # Source and target nodes
            self.assertEqual(graph_edges.shape[1], graph_attn.shape[0])  # Number of edges matches attention values

if __name__ == '__main__':
    unittest.main() 