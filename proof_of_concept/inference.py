
import argparse
import sys
import torch
import numpy as np
import pickle 
from models.training import train_staged_model, TrainingConfig, ModelConfig
from utils.data_factory import get_data, get_available_data_types
from utils.visualization import plot_training_results, save_results, print_training_summary
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from proof_of_concept.models.staged import STAGED
from proof_of_concept.utils.graph_constructor import GraphConstructor
from proof_of_concept.utils.simulated_data_processing import retrieve_simulated_data

from proof_of_concept.tests.test_graph_constructor import create_square_grid_data as create_test_data
from proof_of_concept.utils.visualization import visualize_attention_graph, visualize_graph

from models.staged import STAGED
from models.inference_processor import STAGEDProcessor, PredictionOutput, ODEPredictionOutput


@dataclass
class InferenceOutput:
    """Output from STAGED model inference"""
    predictions: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    time_points: List[int] = None
    cell_type_filter: Optional[int] = None
    prediction_mode: str = None
    model_config: Optional[ModelConfig] = None
    genes: List[str] = None

def setup_device(device_arg: str) -> torch.device:
    """Set up the compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    return device
def inference_main():
    parser = argparse.ArgumentParser(description='Perform inference with trained STAGED models for single cell type')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['one_step', 'k_step', 'ode', 'ode_new'],
                       help='Inference prediction mode (should match training mode)')
    
    parser.add_argument('--data', type=str, required=True,
                       choices=get_available_data_types(),
                       help='Type of data to use for inference')
    
    parser.add_argument('--cell_type_id', type=int, default=None,
                       help='ID of the cell type to perform inference on')
    
    # Mode-specific parameters
    parser.add_argument('--k_steps', type=int, default=3,
                       help='Number of steps for k-step prediction')
    
    parser.add_argument('--eval_times', type=str, default="0.0,0.5,1.0,1.5",
                       help='Comma-separated evaluation times for ODE mode')
    
    parser.add_argument('--ode_method', type=str, default='rk4',
                       choices=['euler', 'rk4', 'dopri5', 'adams'],
                       help='ODE integration method')
    
    # Time range for inference
    parser.add_argument('--start_time', type=int, default=None,
                       help='Starting time point for inference (default: model min_time)')
    
    parser.add_argument('--end_time', type=int, default=None,
                       help='Ending time point for inference (default: last available time)')
    
    # Model parameters (should match training)
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension size (must match training)')
    
    parser.add_argument('--num_gat_layers', type=int, default=1,
                       help='Number of GAT layers (must match training)')
    
    parser.add_argument('--num_mlp_layers', type=int, default=3,
                       help='Number of MLP layers (must match training)')
    
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (must match training)')
    
    # Delta parameters (should match training)
    parser.add_argument('--delta_gl', type=int, default=1,
                       help='Gene-ligand time lag')
    
    parser.add_argument('--delta_lr', type=int, default=2,
                       help='Ligand-receptor time lag')
    
    parser.add_argument('--delta_rg', type=int, default=1,
                       help='Receptor-gene time lag')
    
    parser.add_argument('--delta_gg', type=int, default=0,
                       help='Gene-gene time lag')
    
    # Training config parameters
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference')
    
    parser.add_argument('--distance_threshold', type=float, default=10.0,
                       help='Distance threshold for spatial interactions')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Directory to save inference results')
    
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for inference')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations of predictions')
    
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save raw predictions to file')
    
    parser.add_argument('--store_attention', action='store_true',
                       help='Store attention weights during inference')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load data
    try:
        data = get_data(args.data, device)
        print(f"Data loaded: {data['gene_expression'].shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return 1
    
    # Validate cell type ID
    num_cell_types = len(torch.unique(data['cell_type_assignments']))
    # if args.cell_type_id >= num_cell_types or args.cell_type_id < 0:
    #     print(f"Error: cell_type_id must be between 0 and {num_cell_types-1}")
    #     return 1
    
    # Load trained model checkpoint
    try:
        print(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Extract configurations from checkpoint if available
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Create config from arguments (matching training structure)
            config = TrainingConfig(
                device=device,
                batch_size=args.batch_size,
                distance_threshold=args.distance_threshold
            )
            config.model_config = ModelConfig(
                hidden_dim=args.hidden_dim,
                num_gat_layers=args.num_gat_layers,
                num_mlp_layers=args.num_mlp_layers,
                dropout=args.dropout,
                delta_gl=args.delta_gl,
                delta_lr=args.delta_lr,
                delta_rg=args.delta_rg,
                delta_gg=args.delta_gg,
            )
        
        print("Configuration loaded successfully!")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Perform inference using the same structure as training
    try:

        inference_output = infer_staged_model(
            data=data,
            genes=data['genes'],
            ligand_receptor_pairs=data['ligand_receptor_pairs'],
            receptor_gene_pairs=data['receptor_gene_pairs'],
            cell_type_assignments=data['cell_type_assignments'],
            prior_grns=data['prior_grns'],
            prediction_mode=args.mode,
            k_steps=args.k_steps if args.mode == 'k_step' else None,
            ode_eval_times=parse_eval_times(args.eval_times, device) if args.mode in ['ode'] else None,
            ode_method=args.ode_method,
            config=config,
            cell_type_filter=args.cell_type_id,
            start_time=args.start_time,
            end_time=args.end_time,
            store_attention=args.store_attention,
            distance_threshold=args.distance_threshold
        )
        
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Inference failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    if args.save_predictions:
        save_inference_results(inference_output, args, args.output_dir)
    
    # Create visualizations
    if args.visualize:
        create_inference_visualizations(inference_output, args.output_dir)
    
    # Print summary
    print_inference_summary(inference_output, args)
    
    print(f"\nInference completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    return 0


def infer_staged_model(
    data: Dict[str, torch.Tensor],
    genes: List[str],
    ligand_receptor_pairs: List[tuple],
    receptor_gene_pairs: List[tuple],
    cell_type_assignments: Any,
    prior_grns: Dict[Any, Any],
    prediction_mode: str = "one_step",
    k_steps: Optional[int] = None,
    ode_eval_times: Optional[torch.Tensor] = None,
    ode_method: str = 'rk4',
    distance_threshold: Optional[float] = None,
    config: Optional[TrainingConfig] = None,
    cell_type_filter: Optional[int] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    store_attention: bool = False,
    model_path: str = "results/model.pt",
) -> InferenceOutput:
    """
    Perform inference with a trained STAGED model using the specified prediction mode.
    
    Args:
        data: Dictionary containing:
            - gene_expression: Tensor of shape (n_time_points, n_cells, n_genes)
            - cell_positions: Tensor of shape (n_time_points, n_cells, 2)
            - n_cells: Number of cells
        genes: List of gene identifiers
        ligand_receptor_pairs: List of (ligand, receptor) pairs
        receptor_gene_pairs: List of (receptor, gene) pairs
        cell_type_assignments: Cell type assignments
        prior_grns: Dictionary of prior GRNs
        prediction_mode: One of ["one_step", "k_step", "ode", "ode_new"]
        k_steps: Number of steps for k-step prediction (required if mode is "k_step")
        ode_eval_times: Times to evaluate ODE at (required if mode is "ode")
        ode_method: ODE integration method for ODE mode
        config: Training configuration
        model_checkpoint: Loaded model checkpoint
        cell_type_filter: Specific cell type ID to filter for
        start_time: Starting time point for inference
        end_time: Ending time point for inference
        store_attention: Whether to store attention weights
        
    Returns:
        InferenceOutput containing predictions and metadata
    """
    # Setup configuration (same as training)
    if config is None:
        config = TrainingConfig()
    if config.model_config is None:
        config.model_config = ModelConfig()
    device = config.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Validate ODE mode parameters (same as training)
    if prediction_mode == "ode":
        if ode_eval_times is None:
            raise ValueError("ode_eval_times must be provided for ODE prediction mode")
        ode_eval_times = ode_eval_times.to(device)

    # Initialize model (same structure as training)
    model = STAGED(
        num_genes=len(genes),
        hidden_dim=config.model_config.hidden_dim,
        num_gat_layers=config.model_config.num_gat_layers,
        num_mlp_layers=config.model_config.num_mlp_layers,
        dropout=config.model_config.dropout,
        delta_gl=config.model_config.delta_gl,
        delta_lr=config.model_config.delta_lr,
        delta_rg=config.model_config.delta_rg,
        delta_gg=config.model_config.delta_gg,
        add_self_loops=config.model_config.add_self_loops,
    ).to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    # print(config.distance_threshold)
    # Initialize processor (same as training)
    processor = STAGEDProcessor(
        model=model,
        genes=genes,
        ligand_receptor_pairs=ligand_receptor_pairs,
        receptor_gene_pairs=receptor_gene_pairs,
        cell_type_assignments=cell_type_assignments,
        prior_grns=prior_grns,
        batch_size=config.batch_size,
        distance_threshold=distance_threshold,
        device=device
    )
    
    # Setup ODE if needed (same as training)
    if prediction_mode == "ode":
        processor.setup_ode(data['gene_expression'])
    
    # Get minimum time point based on model lags (same as training)
    min_time = max(model.delta_gl, model.delta_lr, model.delta_rg, model.delta_gg)
    
    # Set time range for inference
    if start_time is None:
        start_time = min_time
    if end_time is None:
        end_time = data['gene_expression'].shape[0]
    
    # Validate k_steps if in k_step mode (same as training)
    if prediction_mode == "k_step":
        if k_steps is None:
            raise ValueError("k_steps must be provided for k_step prediction mode")
        if k_steps >= end_time - start_time:
            raise ValueError(
                f"k_steps ({k_steps}) must be less than available prediction steps "
                f"({end_time - start_time})"
            )
    
    # Filter data for specific cell type if requested
    if cell_type_filter is not None:
        data = filter_data_by_cell_type(data, cell_type_filter)
    
    # Inference loop (similar structure to training loop)
    all_predictions = []
    all_attention_weights = [] if store_attention else None
    
    print(f"Starting inference from time {start_time} to {end_time}...")
    
    with torch.no_grad():  # No gradients needed for inference
        if prediction_mode == "ode":
            # ODE inference mode (similar to training ODE mode)
            for start_t in range(start_time, end_time - len(ode_eval_times) + 1):
                # Set up time range for this segment
                initial_time = float(start_t)
                eval_times_segment = ode_eval_times + start_t
                
                # Get ODE predictions
                ode_output = processor.predict_ode(
                    data=data,
                    initial_time=initial_time,
                    eval_times=eval_times_segment,
                    method=ode_method,
                    store_attention=store_attention
                )
                
                all_predictions.append(ode_output.predictions)
                if store_attention and ode_output.attention_weights is not None:
                    all_attention_weights.append(ode_output.attention_weights)
        
        elif prediction_mode == "ode_new":
            # ODE new inference mode (similar to training ode_new mode)
            for t in range(start_time, end_time):
                print(t)
                # Get ODE predictions for this single timepoint
                ode_output = processor.predict_ode_new(
                    data=data,
                    time_point=t,
                    method=ode_method,
                    store_attention=store_attention
                )
                
                all_predictions.append(ode_output.predictions[0])  # Take first prediction
                if store_attention and ode_output.attention_weights is not None:
                    all_attention_weights.append(ode_output.attention_weights)
        
        else:
            # Original inference modes (one_step, k_step, etc.) - similar to training
            for t in range(start_time, end_time):
                # Get predictions for current time point
                output = processor.predict(
                    data=data,
                    time_point=t,
                    store_attention=store_attention
                )
                
                if prediction_mode == "one_step":
                    # Store one-step prediction
                    all_predictions.append(output.predictions)
                    if store_attention and output.attention_weights is not None:
                        all_attention_weights.append(output.attention_weights)
                else:
                    raise NotImplementedError(
                        f"Prediction mode {prediction_mode} not yet implemented for inference"
                    )
    
    # Stack predictions
    if all_predictions:
        predictions = torch.stack(all_predictions, dim=0)
        if all_attention_weights:
            attention_weights = torch.stack(all_attention_weights, dim=0)
        else:
            attention_weights = None
    else:
        raise ValueError("No predictions generated")
    
    print(predictions.shape)
    # Create inference output
    inference_output = InferenceOutput(
        predictions=predictions,
        attention_weights=attention_weights,
        time_points=list(range(start_time, end_time)),
        cell_type_filter=cell_type_filter,
        prediction_mode=prediction_mode,
        model_config=config.model_config,
        genes=genes
    )
    
    return inference_output


def parse_eval_times(eval_times_str: str, device: torch.device) -> torch.Tensor:
    """Parse evaluation times string into tensor"""
    times = [float(t.strip()) for t in eval_times_str.split(',')]
    return torch.tensor(times, device=device, dtype=torch.float)


def filter_data_by_cell_type(data: Dict[str, torch.Tensor], cell_type_id: int) -> Dict[str, torch.Tensor]:
    """Filter data dictionary to include only specified cell type"""
    cell_type_mask = data['cell_type_assignments'] == cell_type_id
    
    filtered_data = data.copy()
    filtered_data['gene_expression'] = data['gene_expression'][:, cell_type_mask, :]
    filtered_data['cell_positions'] = data['cell_positions'][:, cell_type_mask, :]
    filtered_data['cell_type_assignments'] = data['cell_type_assignments'][cell_type_mask]
    filtered_data['n_cells'] = torch.sum(cell_type_mask).item()
    
    return filtered_data

def save_inference_results(inference_output: InferenceOutput, args, output_dir: str):
    """Save inference results to files"""
    results = {
        'predictions': inference_output.predictions.cpu().numpy(),
        'time_points': inference_output.time_points,
        'cell_type_filter': inference_output.cell_type_filter,
        'prediction_mode': inference_output.prediction_mode,
        'genes': inference_output.genes,
        'model_config': inference_output.model_config,
        'args': vars(args)
    }
    
    if inference_output.attention_weights is not None:
        results['attention_weights'] = inference_output.attention_weights.cpu().numpy()
    
    output_file = os.path.join(output_dir, f'inference_celltype_{args.cell_type_id}_{args.mode}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Predictions saved to: {output_file}")


def create_inference_visualizations(inference_output: InferenceOutput, output_dir: str):
    """Create visualizations of inference results"""
    import matplotlib.pyplot as plt
    
    predictions = inference_output.predictions.cpu().numpy()
    time_points = inference_output.time_points
    
    # Plot gene expression trajectories for top varying genes
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    mean_expr = np.mean(predictions, axis=1)  # Average across cells
    gene_variance = np.var(mean_expr, axis=0)
    top_genes = np.argsort(gene_variance)[-4:]  # Top 4 most variable genes
    
    for i, gene_idx in enumerate(top_genes):
        ax = axes[i//2, i%2]
        ax.plot(time_points, mean_expr[:, gene_idx], 'b-', linewidth=2)
        ax.fill_between(
            time_points,
            mean_expr[:, gene_idx] - np.std(predictions[:, :, gene_idx], axis=1),
            mean_expr[:, gene_idx] + np.std(predictions[:, :, gene_idx], axis=1),
            alpha=0.3
        )
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Expression Level')
        gene_name = inference_output.genes[gene_idx] if inference_output.genes else f'Gene_{gene_idx}'
        ax.set_title(f'{gene_name}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_trajectories.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap of expression changes
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(mean_expr.T, aspect='auto', cmap='viridis')
    ax.set_xlabel('Time Point')
    ax.set_ylabel('Gene Index')
    ax.set_title(f'Gene Expression Heatmap - Cell Type {inference_output.cell_type_filter}')
    plt.colorbar(im)
    plt.savefig(os.path.join(output_dir, 'expression_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def print_inference_summary(inference_output: InferenceOutput, args):
    """Print summary of inference results"""
    predictions = inference_output.predictions
    
    print(f"\n=== Inference Summary ===")
    print(f"Prediction Mode: {inference_output.prediction_mode}")
    print(f"Cell Type Filter: {inference_output.cell_type_filter}")
    print(f"Time Range: {inference_output.time_points[0]} - {inference_output.time_points[-1]}")
    print(f"Number of time points: {len(inference_output.time_points)}")
    print(f"Number of cells: {predictions.shape[1]}")
    print(f"Number of genes: {predictions.shape[2]}")
    
    # Expression statistics
    mean_expr = torch.mean(predictions).item()
    std_expr = torch.std(predictions).item()
    min_expr = torch.min(predictions).item()
    max_expr = torch.max(predictions).item()
    
    print(f"\nExpression Statistics:")
    print(f"Mean expression: {mean_expr:.4f}")
    print(f"Std expression: {std_expr:.4f}")
    print(f"Expression range: [{min_expr:.4f}, {max_expr:.4f}]")
    
    if inference_output.attention_weights is not None:
        print(f"Attention weights stored: {inference_output.attention_weights.shape}")


if __name__ == "__main__":
    inference_main()