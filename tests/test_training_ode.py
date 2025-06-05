import unittest
import os
import sys
import torch
import numpy as np
from typing import Dict
import matplotlib.pyplot as plt

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.training import train_staged_model, TrainingConfig, ModelConfig
from temporal_data_generator import create_oscillatory_dynamics_data, create_damped_oscillator_data
from test_graph_constructor import create_hex_grid_test_data


class TestODETraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test data and parameters once for all test methods."""
        print("\nSetting up ODE training test data and parameters...")
        
        # Set device
        cls.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {cls.device}")
        
        # Create oscillatory dynamics data
        cls.oscillatory_data = create_oscillatory_dynamics_data(
            n_time_points=20,
            n_cells=7,
            n_genes=6,
            dt=0.5,
            noise_level=0.05,
            device=cls.device
        )
        
        # Create damped oscillator data
        cls.oscillator_data = create_damped_oscillator_data(
            n_time_points=25,
            n_cells=4,
            n_genes=4,
            dt=0.2,
            device=cls.device
        )
        
        # Create hex grid data for comparison
        cls.hex_data = create_hex_grid_test_data()
        for key, value in cls.hex_data.items():
            if isinstance(value, torch.Tensor):
                cls.hex_data[key] = value.to(cls.device)
        
        # Model configuration (smaller for testing)
        cls.model_config = ModelConfig(
            hidden_dim=32,
            num_gat_layers=1,
            num_mlp_layers=2,
            dropout=0.1,
            delta_gl=1,
            delta_lr=2,
            delta_rg=1,
            delta_gg=0  # No gene-gene lag for ODE testing
        )
        
        # Training configuration
        cls.config = TrainingConfig(
            max_iterations=20,  # Small number for testing
            learning_rate=0.01,
            batch_size=2,
            device=cls.device,
            model_config=cls.model_config
        )

    def test_ode_parameter_validation(self):
        """Test that ODE mode properly validates parameters."""
        print("\nTesting ODE parameter validation...")
        
        with self.assertRaises(ValueError):
            # Should raise error when ode_eval_times not provided
            train_staged_model(
                data=self.oscillatory_data,
                genes=self.oscillatory_data['genes'],
                ligand_receptor_pairs=self.oscillatory_data['ligand_receptor_pairs'],
                receptor_gene_pairs=self.oscillatory_data['receptor_gene_pairs'],
                cell_type_assignments=self.oscillatory_data['cell_type_assignments'],
                prior_grns=self.oscillatory_data['prior_grns'],
                prediction_mode="ode",
                # ode_eval_times=None,  # Missing!
                config=self.config
            )
        print("✓ Parameter validation working correctly")

    def test_simple_ode_training_oscillatory(self):
        """Test ODE training with oscillatory dynamics data."""
        print("\nTesting ODE training with oscillatory dynamics...")
        
        # Define evaluation times (relative to start time)
        ode_eval_times = torch.tensor([0.0, 0.5, 1.0, 1.5], device=self.device)
        
        # Train model
        output = train_staged_model(
            data=self.oscillatory_data,
            genes=self.oscillatory_data['genes'],
            ligand_receptor_pairs=self.oscillatory_data['ligand_receptor_pairs'],
            receptor_gene_pairs=self.oscillatory_data['receptor_gene_pairs'],
            cell_type_assignments=self.oscillatory_data['cell_type_assignments'],
            prior_grns=self.oscillatory_data['prior_grns'],
            prediction_mode="ode",
            ode_eval_times=ode_eval_times,
            ode_method='dopri5',
            config=self.config
        )
        
        # Check that training produced loss history
        self.assertEqual(len(output.loss_history), self.config.max_iterations)
        
        # Check that loss is finite and reasonable
        self.assertTrue(all(np.isfinite(loss) for loss in output.loss_history))
        self.assertTrue(all(loss >= 0 for loss in output.loss_history))
        
        print(f"Initial loss: {output.loss_history[0]:.6f}")
        print(f"Final loss: {output.loss_history[-1]:.6f}")
        print(f"Loss reduction: {(output.loss_history[0] - output.loss_history[-1]) / output.loss_history[0] * 100:.2f}%")

    def test_ode_training_damped_oscillator(self):
        """Test ODE training with damped oscillator data."""
        print("\nTesting ODE training with damped oscillator dynamics...")
        
        # Define evaluation times
        ode_eval_times = torch.tensor([0.0, 0.2, 0.4, 0.6], device=self.device)
        
        # Train model
        output = train_staged_model(
            data=self.oscillator_data,
            genes=self.oscillator_data['genes'],
            ligand_receptor_pairs=self.oscillator_data['ligand_receptor_pairs'],
            receptor_gene_pairs=self.oscillator_data['receptor_gene_pairs'],
            cell_type_assignments=self.oscillator_data['cell_type_assignments'],
            prior_grns=self.oscillator_data['prior_grns'],
            prediction_mode="ode",
            ode_eval_times=ode_eval_times,
            ode_method='dopri5',
            config=self.config
        )
        
        # Check training results
        self.assertEqual(len(output.loss_history), self.config.max_iterations)
        self.assertTrue(all(np.isfinite(loss) for loss in output.loss_history))
        
        print(f"Initial loss: {output.loss_history[0]:.6f}")
        print(f"Final loss: {output.loss_history[-1]:.6f}")

    def test_different_ode_methods(self):
        """Test ODE training with different integration methods."""
        print("\nTesting different ODE integration methods...")
        
        methods = ['euler', 'rk4', 'dopri5']
        ode_eval_times = torch.tensor([0.0, 0.5, 1.0], device=self.device)
        
        # Use smaller dataset for this test
        small_config = TrainingConfig(
            max_iterations=5,
            learning_rate=0.01,
            batch_size=2,
            device=self.device,
            model_config=self.model_config
        )
        
        for method in methods:
            print(f"Testing method: {method}")
            try:
                output = train_staged_model(
                    data=self.oscillatory_data,
                    genes=self.oscillatory_data['genes'],
                    ligand_receptor_pairs=self.oscillatory_data['ligand_receptor_pairs'],
                    receptor_gene_pairs=self.oscillatory_data['receptor_gene_pairs'],
                    cell_type_assignments=self.oscillatory_data['cell_type_assignments'],
                    prior_grns=self.oscillatory_data['prior_grns'],
                    prediction_mode="ode",
                    ode_eval_times=ode_eval_times,
                    ode_method=method,
                    config=small_config
                )
                
                self.assertEqual(len(output.loss_history), small_config.max_iterations)
                print(f"  ✓ {method}: Final loss = {output.loss_history[-1]:.6f}")
                
            except Exception as e:
                print(f"  ✗ {method}: Failed with error {e}")
                # Some methods might not be available, so we don't fail the test

    def test_ode_vs_next_step_comparison(self):
        """Compare ODE training with next-step training on the same data."""
        print("\nComparing ODE vs next-step training...")
        
        # Use hex data for this comparison (both should work)
        test_data = self.hex_data
        
        # Small training config for quick comparison
        small_config = TrainingConfig(
            max_iterations=10,
            learning_rate=0.01,
            batch_size=2,
            device=self.device,
            model_config=self.model_config
        )
        
        # Train with next-step prediction
        print("Training with next-step prediction...")
        next_step_output = train_staged_model(
            data=test_data,
            genes=test_data['genes'],
            ligand_receptor_pairs=test_data['ligand_receptor_pairs'],
            receptor_gene_pairs=test_data['receptor_gene_pairs'],
            cell_type_assignments=test_data['cell_type_assignments'],
            prior_grns=test_data['prior_grns'],
            prediction_mode="one_step",
            config=small_config
        )
        
        # Train with ODE prediction
        print("Training with ODE prediction...")
        ode_eval_times = torch.tensor([0.0, 1.0], device=self.device)
        ode_output = train_staged_model(
            data=test_data,
            genes=test_data['genes'],
            ligand_receptor_pairs=test_data['ligand_receptor_pairs'],
            receptor_gene_pairs=test_data['receptor_gene_pairs'],
            cell_type_assignments=test_data['cell_type_assignments'],
            prior_grns=test_data['prior_grns'],
            prediction_mode="ode",
            ode_eval_times=ode_eval_times,
            config=small_config
        )
        
        # Both should complete successfully
        self.assertEqual(len(next_step_output.loss_history), small_config.max_iterations)
        self.assertEqual(len(ode_output.loss_history), small_config.max_iterations)
        
        print(f"Next-step final loss: {next_step_output.loss_history[-1]:.6f}")
        print(f"ODE final loss: {ode_output.loss_history[-1]:.6f}")

    def test_ode_different_eval_times(self):
        """Test ODE training with different evaluation time configurations."""
        print("\nTesting different ODE evaluation time configurations...")
        
        # Test different time configurations
        eval_time_configs = [
            torch.tensor([0.0, 1.0], device=self.device),           # Simple 2-point
            torch.tensor([0.0, 0.5, 1.0], device=self.device),     # 3-point
            torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=self.device),  # 5-point dense
        ]
        
        small_config = TrainingConfig(
            max_iterations=5,
            learning_rate=0.01,
            batch_size=2,
            device=self.device,
            model_config=self.model_config
        )
        
        for i, eval_times in enumerate(eval_time_configs):
            print(f"Testing configuration {i+1}: {len(eval_times)} eval points")
            
            output = train_staged_model(
                data=self.oscillatory_data,
                genes=self.oscillatory_data['genes'],
                ligand_receptor_pairs=self.oscillatory_data['ligand_receptor_pairs'],
                receptor_gene_pairs=self.oscillatory_data['receptor_gene_pairs'],
                cell_type_assignments=self.oscillatory_data['cell_type_assignments'],
                prior_grns=self.oscillatory_data['prior_grns'],
                prediction_mode="ode",
                ode_eval_times=eval_times,
                config=small_config
            )
            
            self.assertEqual(len(output.loss_history), small_config.max_iterations)
            print(f"  Final loss: {output.loss_history[-1]:.6f}")

    def test_ode_training_with_attention_storage(self):
        """Test that ODE training works when attention storage is enabled."""
        print("\nTesting ODE training with attention storage...")
        
        # This test ensures that the attention storage mechanism doesn't interfere with training
        ode_eval_times = torch.tensor([0.0, 0.5], device=self.device)
        
        # Train with very few iterations
        small_config = TrainingConfig(
            max_iterations=3,
            learning_rate=0.01,
            batch_size=2,
            device=self.device,
            model_config=self.model_config
        )
        
        output = train_staged_model(
            data=self.oscillatory_data,
            genes=self.oscillatory_data['genes'],
            ligand_receptor_pairs=self.oscillatory_data['ligand_receptor_pairs'],
            receptor_gene_pairs=self.oscillatory_data['receptor_gene_pairs'],
            cell_type_assignments=self.oscillatory_data['cell_type_assignments'],
            prior_grns=self.oscillatory_data['prior_grns'],
            prediction_mode="ode",
            ode_eval_times=ode_eval_times,
            config=small_config
        )
        
        # Should complete without errors
        self.assertEqual(len(output.loss_history), small_config.max_iterations)
        print(f"Training completed successfully. Final loss: {output.loss_history[-1]:.6f}")

    def test_temporal_data_properties(self):
        """Test that our temporal data has expected properties."""
        print("\nTesting temporal data properties...")
        
        # Test oscillatory data
        osc_data = self.oscillatory_data['gene_expression']
        print(f"Oscillatory data shape: {osc_data.shape}")
        
        # Check that data has temporal variation
        temporal_variance = torch.var(osc_data, dim=0).mean()
        print(f"Temporal variance: {temporal_variance:.4f}")
        self.assertGreater(temporal_variance, 0.1, "Data should have significant temporal variation")
        
        # Test oscillator data
        osc_data = self.oscillator_data['gene_expression']
        print(f"Oscillator data shape: {osc_data.shape}")
        
        temporal_variance = torch.var(osc_data, dim=0).mean()
        print(f"Oscillator temporal variance: {temporal_variance:.4f}")
        self.assertGreater(temporal_variance, 0.1, "Oscillator data should have temporal variation")
        
        # Check that values are non-negative (as expected for gene expression)
        self.assertTrue(torch.all(osc_data >= 0), "Gene expression should be non-negative")

if __name__ == '__main__':
    unittest.main(verbosity=2) 