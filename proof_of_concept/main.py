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