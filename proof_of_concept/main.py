"""
THIS IS A PLACEHOLDER FILE. NEED TO REWRITE COMPLETELY.
"""
import argparse
import os
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from time import time

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


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    
    gene_expression_data, genes = load_gene_expression_data(args.expression_data)
    cell_positions = load_cell_positions(args.positions_data)
    ligand_receptor_pairs = load_ligand_receptor_pairs(args.lr_pairs_data)
    

    ##TODO: We should change this to pass the paths as parameters, not the preprocessing pipeline
    simulated_data = retrieve_simulated_data()
    

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
    
    print("Done!")


if __name__ == "__main__":
    main() 