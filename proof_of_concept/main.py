#!/usr/bin/env python3
"""
STAGED Model Training Main Interface

This script provides a command-line interface for training STAGED models
with different prediction modes and data types.

Usage:
    python main.py --mode ode --data oscillatory --iterations 50
    python main.py --mode one_step --data hex_grid --iterations 100
    python main.py --mode ode --data damped_oscillator --eval_times "0.0,0.2,0.4,0.6"
"""

import argparse
import sys
import torch
import numpy as np

from models.training import train_staged_model, TrainingConfig, ModelConfig
from utils.data_factory import get_data, get_available_data_types
from utils.visualization import plot_training_results, save_results, print_training_summary


def parse_eval_times(eval_times_str: str, device: torch.device) -> torch.Tensor:
    """Parse evaluation times from string."""
    try:
        times = [float(x.strip()) for x in eval_times_str.split(',')]
        return torch.tensor(times, device=device)
    except:
        raise ValueError(f"Could not parse evaluation times: {eval_times_str}")
from utils.data_utils import (
    load_gene_expression_data,
    load_cell_positions,
    load_ligand_receptor_pairs,
    load_cell_type_assignments,
    load_prior_grns,
    preprocess_data
)
from utils.visualization import (
    plot_gene_trajectories,
    plot_spatial_expression,
    animate_gene_expression,
    plot_attention_weights,
    plot_training_curves,
    plot_gene_correlations
)
# from trainer import STAGEDTrainer
from utils.graph_constructor import GraphConstructor
from utils.simulated_data_processing import retrieve_simulated_data
from models.training import train_staged_model, TrainingConfig, ModelConfig
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='STAGED: Spatiotemporal Analysis of Gene Expression Dynamics')
    
    # Data paths
    parser.add_argument('--expression_data', type=str, default=None,
                       help='Path to gene expression data file')
    parser.add_argument('--positions_data', type=str, default=None,
                       help='Path to cell position data file')
    parser.add_argument('--lr_pairs_data', type=str, default=None,
                       help='Path to ligand-receptor pairs data file')
    parser.add_argument('--cell_types_data', type=str, default=None,
                       help='Path to cell type assignments data file')
    parser.add_argument('--prior_grns_data', type=str, default=None,
                       help='Path to prior GRNs data file')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='Hidden dimension for the model')
    parser.add_argument('--num_gat_layers', type=int, default=1,
                       help='Number of GAT layers')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                       help='Number of MLP layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout')
    
    # Time lags
    parser.add_argument('--delta_gl', type=int, default=1,
                       help='Time lag for gene -> ligand')
    parser.add_argument('--delta_lr', type=int, default=1,
                       help='Time lag for ligand -> receptor')
    parser.add_argument('--delta_rg', type=int, default=1,
                       help='Time lag for receptor -> gene')
    parser.add_argument('--delta_gg', type=int, default=1,
                       help='Time lag for gene -> gene')
    
    # Training parametersmax_iterations

    parser.add_argument('--max_iterations', type=int, default=10,
                       help='Maximum number of training iterations')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Patience for early stopping')
    parser.add_argument('--validation_fraction', type=float, default=0.2,
                       help='Fraction of training time points to use for validation')
    
    # Spatial parameters
    parser.add_argument('--distance_threshold', type=float, default=10.0,
                       help='Maximum distance to consider cells as neighbors')
    
    # Visualization
    parser.add_argument('--visualize', action='store_false',
                       help='Visualize results')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results and visualizations')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run the model on')
    
    # Add a new argument for time split
    parser.add_argument('--train_end_time', type=int, default=None,
                       help='Time point to end training (later points used for testing)')
    
    return parser.parse_args()


def setup_device(device_arg: str) -> torch.device:
    """Set up the compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device


def print_configuration(args, eval_times=None):
    """Print training configuration summary."""
    print(f"\nTraining Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Data: {args.data}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Delta parameters: GL={args.delta_gl}, LR={args.delta_lr}, RG={args.delta_rg}, GG={args.delta_gg}")
    
    if args.mode == 'k_step':
        print(f"  K steps: {args.k_steps}")
    elif args.mode == 'ode' and eval_times is not None:
        print(f"  Eval times: {eval_times.tolist()}")
        print(f"  ODE method: {args.ode_method}")


def validate_arguments(args):
    """Validate command line arguments."""
    if args.mode == 'ode' and args.delta_gg != 0:
        print("WARNING: delta_gg should be 0 for ODE mode. Setting to 0.")
        args.delta_gg = 0


def create_configurations(args, device):
    """Create model and training configurations."""
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_gat_layers=args.gat_layers,
        num_mlp_layers=args.mlp_layers,
        dropout=args.dropout,
        delta_gl=args.delta_gl,
        delta_lr=args.delta_lr,
        delta_rg=args.delta_rg,
        delta_gg=args.delta_gg
    )
    
    training_config = TrainingConfig(
        max_iterations=args.iterations,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=device,
        model_config=model_config
    )
    
    return model_config, training_config


def prepare_mode_kwargs(args, device):
    """Prepare mode-specific keyword arguments."""
    mode_kwargs = {}
    eval_times = None
    
    if args.mode == 'k_step':
        mode_kwargs['k_steps'] = args.k_steps
    
    elif args.mode == 'ode':
        eval_times = parse_eval_times(args.eval_times, device)
        mode_kwargs['ode_eval_times'] = eval_times
        mode_kwargs['ode_method'] = args.ode_method
    # Load data
    print("Loading data...")
    
    gene_expression_data, genes = load_gene_expression_data(args.expression_data)
    cell_positions = load_cell_positions(args.positions_data)
    ligand_receptor_pairs = load_ligand_receptor_pairs(args.lr_pairs_data)
    

    ##TODO: We should change this to pass the paths as parameters, not the preprocessing pipeline
    simulated_data = retrieve_simulated_data(data_dir="data/raw",sim_file="100_simulation_results.pkl")
    

    # Model configuration
    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,  # Smaller for testing
        num_gat_layers=args.num_gat_layers,
        num_mlp_layers=args.num_mlp_layers,
        dropout=args.dropout
    )
    
    # Training configuration
    config = TrainingConfig(
        max_iterations=args.max_iterations,  # Small number for testing
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
        output_model_path=os.path.join(args.output_dir, "model.pt"),
        model_config=model_config
    )

    results = train_staged_model(
        data=simulated_data,
        genes=simulated_data['genes'],
        ligand_receptor_pairs=simulated_data['ligand_receptor_pairs'],
        receptor_gene_pairs=simulated_data['receptor_gene_pairs'],
        cell_type_assignments=simulated_data['cell_type_assignments'],
        prior_grns=simulated_data['prior_grns'],
        prediction_mode="one_step",
        config=config
    )
    
    
    # Check that loss decreased
    print(f"Initial loss: {results.loss_history[0]:.6f}")
    print(f"Final loss: {results.loss_history[-1]:.6f}")
    
    # Save results object to output directory
    results_path = os.path.join(args.output_dir, "results.pkl")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_path}")

    # Save config for future model loading
    config_path = os.path.join(args.output_dir, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(config.__dict__, f)
    print(f"Config saved to {config_path}")
    
    # # Visualize results if requested
    # if args.visualize:
    #     print("Generating visualizations...")
        
    #     # Set the output directory for figures
    #     os.chdir(args.output_dir)
        
    #     # Plot training curves
    #     plot_training_curves(results)
        
    #     # Plot gene trajectories for a sample of cells
    #     sample_cells = np.random.choice(cell_ids, min(5, len(cell_ids)), replace=False)
    #     sample_genes = np.random.choice(range(len(genes)), min(5, len(genes)), replace=False)
        
    #     for cell_id in sample_cells:
    #         plot_gene_trajectories(gene_expression_data, predictions, cell_id, sample_genes)
        
    #     # Plot spatial expression for a sample gene and time point
    #     sample_gene = np.random.choice(range(len(genes)))
        
    #     # Get the maximum time point
    #     max_time = max(t for cell_id in gene_expression_data for gene_idx in gene_expression_data[cell_id] 
    #                   for t in gene_expression_data[cell_id][gene_idx].keys())
        
    #     for t in range(0, max_time + 1, 2):  # Plot every other time point
    #         plot_spatial_expression(cell_positions, gene_expression_data, t, sample_gene)
        
    #     # Create an animation for a sample gene
    #     try:
    #         animate_gene_expression(cell_positions, gene_expression_data, sample_gene)
    #     except Exception as e:
    #         print(f"Warning: Could not create animation due to: {e}")
        
    #     # Plot gene correlations at a sample time point
    #     plot_gene_correlations(gene_expression_data, cell_ids, sample_genes, max_time // 2)
        
    #     # Add visualization for time-based prediction evaluation
    #     if 'test_time_points' in results and results['test_time_points']:
    #         print("Generating time-based prediction visualizations...")
            
    #         # Create a directory for time-split results
    #         time_split_dir = os.path.join(args.output_dir, 'time_split_results')
    #         os.makedirs(time_split_dir, exist_ok=True)
    #         os.chdir(time_split_dir)
            
    #         # Plot training vs testing time points
    #         plt.figure(figsize=(10, 6))
    #         all_times = results['train_time_points'] + results['test_time_points']
    #         plt.axvline(x=results['train_time_points'][-1], color='r', linestyle='--', 
    #                   label='Train/Test Split')
    #         plt.scatter(results['train_time_points'], 
    #                    [0.5] * len(results['train_time_points']), 
    #                    label='Training', color='blue', s=100)
    #         plt.scatter(results['test_time_points'], 
    #                    [0.5] * len(results['test_time_points']), 
    #                    label='Testing', color='green', s=100)
    #         plt.yticks([])
    #         plt.xlabel('Time Points')
    #         plt.title('Time-Based Train/Test Split')
    #         plt.legend()
    #         plt.savefig('time_split.png')
    #         plt.close()
            
    #         # Calculate prediction error for test time points
    #         test_time_points = results['test_time_points']
    #         if test_time_points:
    #             mse_by_time = {}
    #             mse_by_cell = {}
    #             mse_by_gene = {}
                
    #             for t in test_time_points:
    #                 errors = []
    #                 for cell_id in cell_ids:
    #                     for gene_idx in range(len(genes)):
    #                         if (gene_idx in gene_expression_data[cell_id] and 
    #                             t in gene_expression_data[cell_id][gene_idx] and
    #                             gene_idx in predictions[cell_id] and
    #                             t in predictions[cell_id][gene_idx]):
                                
    #                             actual = gene_expression_data[cell_id][gene_idx][t]
    #                             pred = predictions[cell_id][gene_idx][t]
    #                             error = (actual - pred) ** 2
    #                             errors.append(error)
                                
    #                             # Track error by cell
    #                             if cell_id not in mse_by_cell:
    #                                 mse_by_cell[cell_id] = []
    #                             mse_by_cell[cell_id].append(error)
                                
    #                             # Track error by gene
    #                             if gene_idx not in mse_by_gene:
    #                                 mse_by_gene[gene_idx] = []
    #                             mse_by_gene[gene_idx].append(error)
                    
    #                 mse_by_time[t] = np.mean(errors) if errors else np.nan
                
    #             # Plot MSE by time point
    #             plt.figure(figsize=(10, 6))
    #             time_points = sorted(mse_by_time.keys())
    #             mse_values = [mse_by_time[t] for t in time_points]
    #             plt.plot(time_points, mse_values, 'o-', linewidth=2)
    #             plt.xlabel('Time Point')
    #             plt.ylabel('Mean Squared Error')
    #             plt.title('Prediction Error by Time Point')
    #             plt.grid(True, linestyle='--', alpha=0.6)
    #             plt.savefig('mse_by_time.png')
    #             plt.close()
                
    #             # Plot MSE by cell (top 10 cells)
    #             plt.figure(figsize=(12, 6))
    #             cell_mse = {cell_id: np.mean(errors) for cell_id, errors in mse_by_cell.items()}
    #             top_cells = sorted(cell_mse.items(), key=lambda x: x[1])[:10]
    #             cell_ids_plot = [cell_id for cell_id, _ in top_cells]
    #             mse_values = [cell_mse[cell_id] for cell_id in cell_ids_plot]
    #             plt.bar(cell_ids_plot, mse_values)
    #             plt.xlabel('Cell ID')
    #             plt.ylabel('Mean Squared Error')
    #             plt.title('Prediction Error by Cell (Top 10)')
    #             plt.xticks(rotation=45)
    #             plt.tight_layout()
    #             plt.savefig('mse_by_cell.png')
    #             plt.close()
                
    #             # Plot MSE by gene (top 10 genes)
    #             plt.figure(figsize=(12, 6))
    #             gene_mse = {gene_idx: np.mean(errors) for gene_idx, errors in mse_by_gene.items()}
    #             top_genes = sorted(gene_mse.items(), key=lambda x: x[1])[:10]
    #             gene_indices_plot = [gene_idx for gene_idx, _ in top_genes]
    #             gene_names = [genes[idx] for idx in gene_indices_plot]
    #             mse_values = [gene_mse[gene_idx] for gene_idx in gene_indices_plot]
    #             plt.bar(gene_names, mse_values)
    #             plt.xlabel('Gene')
    #             plt.ylabel('Mean Squared Error')
    #             plt.title('Prediction Error by Gene (Top 10)')
    #             plt.xticks(rotation=45)
    #             plt.tight_layout()
    #             plt.savefig('mse_by_gene.png')
    #             plt.close()
                
    #             # Plot actual vs predicted for a few selected cells and genes
    #             sample_cells = np.random.choice(cell_ids, min(3, len(cell_ids)), replace=False)
    #             sample_genes = np.random.choice(range(len(genes)), min(3, len(genes)), replace=False)
                
    #             for cell_id in sample_cells:
    #                 for gene_idx in sample_genes:
    #                     plt.figure(figsize=(10, 6))
                        
    #                     # Plot training data
    #                     train_times = []
    #                     train_vals = []
    #                     for t in results['train_time_points']:
    #                         if gene_idx in gene_expression_data[cell_id] and t in gene_expression_data[cell_id][gene_idx]:
    #                             train_times.append(t)
    #                             train_vals.append(gene_expression_data[cell_id][gene_idx][t])
                        
    #                     plt.plot(train_times, train_vals, 'bo-', label='Training Data')
                        
    #                     # Plot test data and predictions
    #                     test_times = []
    #                     test_vals = []
    #                     pred_times = []
    #                     pred_vals = []
                        
    #                     for t in results['test_time_points']:
    #                         if gene_idx in gene_expression_data[cell_id] and t in gene_expression_data[cell_id][gene_idx]:
    #                             test_times.append(t)
    #                             test_vals.append(gene_expression_data[cell_id][gene_idx][t])
                            
    #                         if gene_idx in predictions[cell_id] and t in predictions[cell_id][gene_idx]:
    #                             pred_times.append(t)
    #                             pred_vals.append(predictions[cell_id][gene_idx][t])
                        
    #                     plt.plot(test_times, test_vals, 'go-', label='Actual (Test)')
    #                     plt.plot(pred_times, pred_vals, 'ro--', label='Predicted')
                        
    #                     plt.axvline(x=results['train_time_points'][-1], color='k', linestyle='--', 
    #                               label='Train/Test Split')
                        
    #                     plt.title(f'Gene {genes[gene_idx]} Expression in Cell {cell_id}')
    #                     plt.xlabel('Time')
    #                     plt.ylabel('Expression')
    #                     plt.legend()
    #                     plt.grid(True, linestyle='--', alpha=0.6)
                        
    #                     plt.savefig(f'time_prediction_{cell_id}_{genes[gene_idx]}.png')
    #                     plt.close()
            
    #         os.chdir('..')  # Return to output directory
        
    #     print(f"Visualizations saved to {args.output_dir}")
    
    return mode_kwargs, eval_times


def main():
    parser = argparse.ArgumentParser(description='Train STAGED models with different modes and data types')
    
    # Required arguments
    parser.add_argument('--mode', type=str, required=True,
                       choices=['one_step', 'k_step', 'ode'],
                       help='Training prediction mode')
    
    parser.add_argument('--data', type=str, required=True,
                       choices=get_available_data_types(),
                       help='Type of data to use for training')
    
    # Training parameters
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of training iterations')
    
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension size')
    
    parser.add_argument('--gat_layers', type=int, default=1,
                       help='Number of GAT layers (must be 1 for current STAGED implementation)')
    
    parser.add_argument('--mlp_layers', type=int, default=3,
                       help='Number of MLP layers')
    
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Mode-specific parameters
    parser.add_argument('--k_steps', type=int, default=3,
                       help='Number of steps for k-step prediction')
    
    parser.add_argument('--eval_times', type=str, default="0.0,0.5,1.0,1.5",
                       help='Comma-separated evaluation times for ODE mode')
    
    parser.add_argument('--ode_method', type=str, default='dopri5',
                       choices=['euler', 'rk4', 'dopri5', 'adams'],
                       help='ODE integration method')
    
    # Delta parameters
    parser.add_argument('--delta_gl', type=int, default=1,
                       help='Gene-ligand time lag')
    
    parser.add_argument('--delta_lr', type=int, default=2,
                       help='Ligand-receptor time lag')
    
    parser.add_argument('--delta_rg', type=int, default=1,
                       help='Receptor-gene time lag')
    
    parser.add_argument('--delta_gg', type=int, default=0,
                       help='Gene-gene time lag (should be 0 for ODE mode)')
    
    # Output options
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save results (if not specified, results are displayed only)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Validate arguments
    validate_arguments(args)
    
    # Load data
    try:
        data = get_data(args.data, device)
        print(f"Data loaded: {data['gene_expression'].shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Create configurations
    model_config, training_config = create_configurations(args, device)
    
    # Prepare mode-specific arguments
    try:
        mode_kwargs, eval_times = prepare_mode_kwargs(args, device)
    except Exception as e:
        print(f"Error preparing mode arguments: {e}")
        return 1
    
    # Print configuration
    print_configuration(args, eval_times)
    
    # Train model
    print(f"\nStarting training...")
    try:
        output = train_staged_model(
            data=data,
            genes=data['genes'],
            ligand_receptor_pairs=data['ligand_receptor_pairs'],
            receptor_gene_pairs=data['receptor_gene_pairs'],
            cell_type_assignments=data['cell_type_assignments'],
            prior_grns=data['prior_grns'],
            prediction_mode=args.mode,
            config=training_config,
            **mode_kwargs
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Display and save results
    print_training_summary(output, args)
    
    if args.save_dir:
        save_results(output, args, data, args.save_dir)
    else:
        plot_training_results(output)
    
    print("\nTraining completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 