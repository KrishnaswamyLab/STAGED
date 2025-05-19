import unittest
import os
import sys
import torch
import networkx as nx
import numpy as np
import random

# Add the parent directory (proof_of_concept) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.staged import STAGED
from models.inference_processor import STAGEDProcessor, PredictionOutput
from utils.graph_constructor import GraphConstructor
from utils.graph_data_handler import GraphDataHandler
from test_graph_constructor import create_square_grid_data

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TestSTAGEDProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and parameters once for all test methods."""
        print("\nSetting up test data and parameters...")
        # Set random seeds
        set_random_seeds()
        
        # Set up test data
        cls.data = create_square_grid_data()
        
        # Model parameters
        cls.delta_gl = 1
        cls.delta_lr = 5
        cls.delta_rg = 3
        cls.delta_gg = 7
        cls.distance_threshold = 10.0
        cls.batch_size = 10
        cls.time_point = 10
        cls.hidden_dim = 64
        cls.num_gat_layers = 1
        cls.num_mlp_layers = 3
        cls.dropout = 0.0  # Set dropout to 0 for deterministic behavior
        cls.add_self_loops = False

        # Set device
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {cls.device}")

    def setUp(self):
        """Set up test-specific components."""
        print("\nInitializing model components...")
        # Reset random seeds before each test
        set_random_seeds()

        # Initialize original model
        self.model = STAGED(
            num_genes=len(self.data['genes']),
            hidden_dim=self.hidden_dim,
            num_gat_layers=self.num_gat_layers,
            num_mlp_layers=self.num_mlp_layers,
            dropout=self.dropout,
            delta_gl=self.delta_gl,
            delta_lr=self.delta_lr,
            delta_rg=self.delta_rg,
            delta_gg=self.delta_gg,
            add_self_loops=self.add_self_loops,
        ).to(self.device)
        self.model.eval()

        # Initialize graph constructor and handler for reference implementation
        self.graph_constructor = GraphConstructor(
            genes=self.data['genes'],
            ligand_receptor_pairs=self.data['ligand_receptor_pairs'],
            receptor_gene_pairs=self.data['receptor_gene_pairs'],
            cell_type_assignments=self.data['cell_type_assignments'],
            prior_grns=self.data['prior_grns']
        )
        self.graph_handler = GraphDataHandler(self.model, device=self.device)

        # Initialize processor with same model
        self.processor = STAGEDProcessor(
            model=self.model,
            genes=self.data['genes'],
            ligand_receptor_pairs=self.data['ligand_receptor_pairs'],
            receptor_gene_pairs=self.data['receptor_gene_pairs'],
            cell_type_assignments=self.data['cell_type_assignments'],
            prior_grns=self.data['prior_grns'],
            batch_size=self.batch_size,
            distance_threshold=self.distance_threshold,
            device=self.device
        )

        print("Components initialized with identical model and zero dropout")

    def compare_tensors(self, tensor1, tensor2, name1, name2, atol=1e-6):
        """Compare two tensors and print detailed differences."""
        print(f"\nComparing {name1} vs {name2}:")
        
        # Convert to float for comparison if needed
        if tensor1.dtype != torch.float32:
            tensor1 = tensor1.float()
        if tensor2.dtype != torch.float32:
            tensor2 = tensor2.float()
        
        # Check shapes
        if tensor1.shape != tensor2.shape:
            print(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
            return False
        
        # Compute differences
        abs_diff = (tensor1 - tensor2).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        num_different = (abs_diff > atol).sum().item()
        
        print(f"Maximum absolute difference: {max_diff:.6f}")
        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Number of elements differing by more than {atol}: {num_different}")
        
        # If there are differences, print some examples
        if num_different > 0:
            different_indices = (abs_diff > atol).nonzero()
            print("\nExample differences (up to 5):")
            for idx in different_indices[:5]:
                idx_tuple = tuple(idx.cpu().numpy())
                val1 = tensor1[idx_tuple].item()
                val2 = tensor2[idx_tuple].item()
                print(f"At index {idx_tuple}: {val1:.6f} vs {val2:.6f} (diff: {abs(val1-val2):.6f})")
        
        return num_different == 0

    def print_tensor_stats(self, tensor, name):
        """Print detailed statistics about a tensor."""
        print(f"\n{name} statistics:")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        if tensor.dtype in [torch.float32, torch.float64]:
            print(f"Mean: {tensor.mean().item():.6f}")
            print(f"Std: {tensor.std().item():.6f}")
            print(f"Min: {tensor.min().item():.6f}")
            print(f"Max: {tensor.max().item():.6f}")
        if tensor.numel() < 20:  # Print all values if tensor is small
            print(f"All values: {tensor.detach().cpu().numpy()}")

    def test_predictions_match_reference(self):
        """Test if processor predictions match reference implementation."""
        print("\nRunning prediction matching test...")
        
        # --- Run reference implementation ---
        print("Constructing cell graphs for reference implementation...")
        cell_graphs = []
        for cell_idx in range(self.data['n_cells']):
            base_graph = self.graph_constructor.construct_base_graph(cell_idx)
            updated_graph = self.graph_constructor.update_graph_with_neighbors(
                base_graph, cell_idx, self.data['cell_positions'], 
                self.time_point, distance_threshold=self.distance_threshold
            )
            pyg_graph = self.graph_constructor.assign_node_features(
                updated_graph, cell_idx, self.time_point, 
                self.data['gene_expression'],
                self.delta_gl, self.delta_lr, self.delta_rg, self.delta_gg
            )
            cell_graphs.append(pyg_graph.to(self.device))

        print("Getting predictions from reference implementation...")
        with torch.no_grad():
            predictions_ref, attention_ref, node_ptr_ref = self.graph_handler.process_cell_graphs(
                cell_graphs,
                num_genes=len(self.data['genes']),
                batch_size=self.batch_size
            )

        # --- Run processor implementation ---
        print("Getting predictions from processor...")
        module_input_data = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) 
            for k, v in self.data.items()
        }

        output = self.processor.predict(
            data=module_input_data,
            time_point=self.time_point
        )

        # --- Compare results with detailed debugging ---
        print("\nComparing predictions...")
        predictions_match = self.compare_tensors(
            output.predictions, predictions_ref,
            "Processor predictions", "Reference predictions"
        )
        
        if predictions_match:
            print("\n✓ Predictions match successfully!")
        
        # Compare attention weights
        print("\nComparing attention weights...")
        attn_proc_edges, attn_proc_vals = output.attention_weights
        attn_ref_edges, attn_ref_vals = attention_ref
        
        edges_match = self.compare_tensors(
            attn_proc_edges.float(), attn_ref_edges.float(),
            "Processor attention edges", "Reference attention edges"
        )
        
        values_match = self.compare_tensors(
            attn_proc_vals, attn_ref_vals,
            "Processor attention values", "Reference attention values",
            atol=1e-5
        )
        
        if edges_match:
            print("\n✓ Attention edges match successfully!")
        if values_match:
            print("\n✓ Attention values match successfully!")

        # Compare node pointers
        print("\nComparing node pointers...")
        pointers_match = self.compare_tensors(
            output.node_pointers, node_ptr_ref,
            "Processor node pointers", "Reference node pointers"
        )
        
        if pointers_match:
            print("\n✓ Node pointers match successfully!")

        # Final assertions with detailed messages
        self.assertTrue(predictions_match, "Prediction values do not match")
        self.assertTrue(edges_match, "Attention edge indices do not match")
        self.assertTrue(values_match, "Attention values do not match")
        self.assertTrue(pointers_match, "Node pointers do not match")

    def test_predict_specific_cells(self):
        """Test prediction for specific cell subset."""
        print("\nTesting prediction for specific cells...")
        cell_ids = [0, 2]  # Test with first and third cells
        module_input_data = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) 
            for k, v in self.data.items()
        }

        output = self.processor.predict(
            data=module_input_data,
            time_point=self.time_point,
            cell_ids=cell_ids
        )

        expected_shape = (len(cell_ids), len(self.data['genes']))
        actual_shape = output.predictions.shape
        
        print(f"Expected prediction shape: {expected_shape}")
        print(f"Actual prediction shape: {actual_shape}")
        
        shape_matches = actual_shape == expected_shape
        if shape_matches:
            print("\n✓ Specific cells prediction shape matches expected!")
        
        self.assertEqual(actual_shape, expected_shape,
                        f"Expected predictions for {len(cell_ids)} cells, got {output.predictions.shape[0]}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 