import unittest
import os
import sys
import torch
import numpy as np
import networkx as nx

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.data_utils import (
    load_gene_expression_data,
    load_cell_positions,
    load_ligand_receptor_pairs,
    load_cell_type_assignments,
    load_prior_grns
)
from trainer import STAGEDTrainer


class TestIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for integration test"""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create smaller test dataset
        self.num_cells = 5
        self.num_genes = 10
        self.num_time_points = 5
        
        # Create gene expression data
        self.gene_expression_data = {}
        for c in range(self.num_cells):
            cell_id = f"cell_{c}"
            self.gene_expression_data[cell_id] = {}
            for g in range(self.num_genes):
                base = np.random.normal(0, 1)
                trend = np.random.normal(0, 0.1, self.num_time_points)
                expression = base + np.cumsum(trend)
                noise = np.random.normal(0, 0.05, self.num_time_points)
                expression += noise
                self.gene_expression_data[cell_id][g] = {t: float(expression[t]) for t in range(self.num_time_points)}
        
        # Create genes list
        self.genes = [f"gene_{g}" for g in range(self.num_genes)]
        
        # Create cell positions
        self.cell_positions = {}
        for c in range(self.num_cells):
            cell_id = f"cell_{c}"
            start_pos = np.random.uniform(-10, 10, 2)
            movement = np.random.normal(0, 0.5, (self.num_time_points, 2))
            positions = start_pos + np.cumsum(movement, axis=0)
            self.cell_positions[cell_id] = {t: positions[t].tolist() for t in range(self.num_time_points)}
        
        # Create ligand-receptor pairs
        self.lr_pairs = []
        for _ in range(3):  # 3 pairs
            ligand = f"gene_{np.random.randint(0, self.num_genes)}"
            receptor = f"gene_{np.random.randint(0, self.num_genes)}"
            if ligand != receptor:  # Avoid self-loops
                self.lr_pairs.append((ligand, receptor))
        
        # Create cell type assignments
        self.cell_ids = list(self.gene_expression_data.keys())
        self.cell_types = load_cell_type_assignments(None, self.cell_ids)
        self.unique_cell_types = list(set(self.cell_types.values()))
        
        # Create prior GRNs
        self.prior_grns = load_prior_grns(None, self.genes, self.unique_cell_types)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        # Initialize trainer
        trainer = STAGEDTrainer(
            genes=self.genes,
            ligand_receptor_pairs=self.lr_pairs,
            cell_type_assignments=self.cell_types,
            prior_grns=self.prior_grns,
            delta_gl=1,
            delta_lr=1,
            delta_rg=1,
            delta_gg=1
        )
        
        # Check that trainer has been initialized correctly
        self.assertEqual(trainer.num_genes, self.num_genes)
        self.assertEqual(len(trainer.ligand_receptor_pairs), len(self.lr_pairs))
        
        # Check that model has been created
        self.assertTrue(hasattr(trainer, 'model'))
        self.assertEqual(trainer.model.num_genes, self.num_genes)
        
        # Check that graph constructor has been created
        self.assertTrue(hasattr(trainer, 'graph_constructor'))
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_support(self):
        """Test that model can be moved to CUDA if available"""
        if torch.cuda.is_available():
            trainer = STAGEDTrainer(
                genes=self.genes,
                ligand_receptor_pairs=self.lr_pairs,
                cell_type_assignments=self.cell_types,
                prior_grns=self.prior_grns,
                device='cuda'
            )
            
            # Check that model is on CUDA
            self.assertTrue(next(trainer.model.parameters()).is_cuda)
    
    def test_mini_training_run(self):
        """Test a small training run to ensure all components work together"""
        # Initialize trainer
        trainer = STAGEDTrainer(
            genes=self.genes,
            ligand_receptor_pairs=self.lr_pairs,
            cell_type_assignments=self.cell_types,
            prior_grns=self.prior_grns,
            delta_gl=1,
            delta_lr=1,
            delta_rg=1,
            delta_gg=1
        )
        
        # Run a mini training session (2 epochs)
        results = trainer.train(
            gene_expression_data=self.gene_expression_data,
            cell_positions=self.cell_positions,
            num_epochs=2,
            batch_size=2,
            validation_split=0.2,
            patience=5
        )
        
        # Check results structure
        self.assertIn('train_losses', results)
        self.assertIn('val_losses', results)
        self.assertIn('predictions', results)
        
        # Check losses
        self.assertEqual(len(results['train_losses']), 2)  # 2 epochs
        self.assertEqual(len(results['val_losses']), 2)  # 2 epochs
        
        # Check predictions
        self.assertEqual(len(results['predictions']), self.num_cells)
        
        # Check a random cell's predictions
        first_cell = list(results['predictions'].keys())[0]
        self.assertEqual(len(results['predictions'][first_cell]), self.num_genes)
        
        # Check time points in predictions
        # t_init is 1 (max of deltas), so predictions start from t=2
        gene_0_preds = results['predictions'][first_cell][0]
        self.assertIn(2, gene_0_preds)  # Should have prediction for t=2
        self.assertIn(3, gene_0_preds)  # Should have prediction for t=3
        self.assertIn(4, gene_0_preds)  # Should have prediction for t=4


if __name__ == '__main__':
    unittest.main() 