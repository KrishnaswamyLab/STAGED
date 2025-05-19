import unittest
import os
import sys
import numpy as np
import networkx as nx

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_utils import (
    load_gene_expression_data,
    load_cell_positions,
    load_ligand_receptor_pairs,
    load_cell_type_assignments,
    load_prior_grns,
    preprocess_data
)


class TestDataUtils(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        # Set random seed for reproducibility
        np.random.seed(42)
    
    def test_load_gene_expression_data(self):
        """Test loading gene expression data"""
        # Since we don't have actual files, this tests the dummy data generation
        gene_expression_data, genes = load_gene_expression_data(None)
        
        # Check structure
        self.assertIsInstance(gene_expression_data, dict)
        self.assertIsInstance(genes, list)
        
        # Check dimensions
        self.assertEqual(len(gene_expression_data), 100)  # Default 100 cells
        self.assertEqual(len(genes), 50)  # Default 50 genes
        
        # Check a random cell
        first_cell = list(gene_expression_data.keys())[0]
        self.assertEqual(len(gene_expression_data[first_cell]), 50)  # 50 genes
        
        # Check time points
        self.assertEqual(len(gene_expression_data[first_cell][0]), 10)  # 10 time points
    
    def test_load_cell_positions(self):
        """Test loading cell positions"""
        # Since we don't have actual files, this tests the dummy data generation
        cell_positions = load_cell_positions(None)
        
        # Check structure
        self.assertIsInstance(cell_positions, dict)
        
        # Check dimensions
        self.assertEqual(len(cell_positions), 100)  # Default 100 cells
        
        # Check a random cell
        first_cell = list(cell_positions.keys())[0]
        self.assertEqual(len(cell_positions[first_cell]), 10)  # 10 time points
        
        # Check position format
        self.assertEqual(len(cell_positions[first_cell][0]), 2)  # 2D position
    
    def test_load_ligand_receptor_pairs(self):
        """Test loading ligand-receptor pairs"""
        # Since we don't have actual files, this tests the dummy data generation
        lr_pairs = load_ligand_receptor_pairs(None)
        
        # Check structure
        self.assertIsInstance(lr_pairs, list)
        
        # Check dimensions
        self.assertEqual(len(lr_pairs), 20)  # Default 20 pairs
        
        # Check pair format
        for pair in lr_pairs:
            self.assertEqual(len(pair), 2)
            self.assertNotEqual(pair[0], pair[1])  # No self-loops
    
    def test_load_cell_type_assignments(self):
        """Test loading cell type assignments"""
        # Create cell IDs
        cell_ids = [f"cell_{i}" for i in range(10)]
        
        # Since we don't have actual files, this tests the dummy data generation
        cell_types = load_cell_type_assignments(None, cell_ids)
        
        # Check structure
        self.assertIsInstance(cell_types, dict)
        
        # Check dimensions
        self.assertEqual(len(cell_types), 10)  # 10 cells
        
        # Check that all cells have types assigned
        for cell_id in cell_ids:
            self.assertIn(cell_id, cell_types)
            self.assertTrue(cell_types[cell_id].startswith("type_"))
    
    def test_load_prior_grns(self):
        """Test loading prior GRNs"""
        # Create genes and cell types
        genes = [f"gene_{i}" for i in range(5)]
        cell_types = ["type_A", "type_B"]
        
        # Since we don't have actual files, this tests the dummy data generation
        prior_grns = load_prior_grns(None, genes, cell_types)
        
        # Check structure
        self.assertIsInstance(prior_grns, dict)
        
        # Check dimensions
        self.assertEqual(len(prior_grns), 2)  # 2 cell types
        
        # Check that all cell types have GRNs
        for cell_type in cell_types:
            self.assertIn(cell_type, prior_grns)
            self.assertIsInstance(prior_grns[cell_type], nx.DiGraph)
            
            # Check that all genes are in the GRN
            for gene in genes:
                self.assertIn(gene, prior_grns[cell_type].nodes())
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        # Create sample gene expression data
        gene_expression_data = {
            "cell_0": {
                0: {0: 1.0, 1: 2.0},
                1: {0: -1.0, 1: -2.0}
            },
            "cell_1": {
                0: {0: 3.0, 1: 4.0},
                1: {0: -3.0, 1: -4.0}
            }
        }
        
        # Preprocess data with normalization
        processed_data = preprocess_data(gene_expression_data, normalize=True)
        
        # Check structure preservation
        self.assertEqual(len(processed_data), len(gene_expression_data))
        self.assertEqual(len(processed_data["cell_0"]), len(gene_expression_data["cell_0"]))
        self.assertEqual(len(processed_data["cell_0"][0]), len(gene_expression_data["cell_0"][0]))
        
        # Check normalization
        all_values = []
        for cell in processed_data.values():
            for gene in cell.values():
                for value in gene.values():
                    all_values.append(value)
        
        # Standard scaling should result in mean ~0 and std ~1
        self.assertAlmostEqual(np.mean(all_values), 0.0, places=2)
        self.assertAlmostEqual(np.std(all_values), 1.0, places=2)
        
        # Test without normalization
        processed_data_no_norm = preprocess_data(gene_expression_data, normalize=False)
        self.assertEqual(processed_data_no_norm, gene_expression_data)


if __name__ == '__main__':
    unittest.main() 