#!/usr/bin/env python3
"""
STAGED Model Inference Interface

This script provides a command-line interface for running inference with STAGED models.

Usage:
    python inference.py --config config/inference_config.yaml --checkpoint_path path/to/model.pt
"""

import argparse
import torch

from src.config.config import load_config
from src.utils.data_factory import get_data
from trainer.predictor import STAGEDPredictor,print_inference_summary

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='STAGED: Spatiotemporal Analysis of Gene Expression Dynamics - Inference')
    
    # Required arguments
    parser.add_argument('--config', 
                       type=str, 
                       required=True,
                       help='Path to configuration file (YAML format)')
    
    parser.add_argument('--checkpoint_path',
                       type=str,
                       required=True,
                       help='Path to model checkpoint for inference')
    
    # Optional arguments
    parser.add_argument('--device', 
                       type=str, 
                       default=None,
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Override device setting from config')
    
    parser.add_argument('--output_dir', 
                       type=str, 
                       default=None,
                       help='Override output directory from config')
    
    parser.add_argument('--seed', 
                       type=int, 
                       default=None,
                       help='Override random seed from config')
    
    parser.add_argument('--initial_time',
                       type=int,
                       default=None,
                       help='Initial time point for predictions')
    
    parser.add_argument('--prediction_steps',
                       type=int,
                       default=None,
                       help='Number of time steps to predict')
    
    parser.add_argument('--cell_type_id',
                       type=int,
                       default=None,
                       help='ID of the cell type to perform inference on')

    return parser.parse_args()

def setup_environment(config, args):
    """Setup the inference environment."""
    # Override config with command line arguments if provided
    if args.device:
        config.system.device = args.device
    if args.output_dir:
        config.inference.output_dir = args.output_dir
    if args.seed:
        config.system.seed = args.seed
    
    # Set device
    if config.system.device == "auto":
        config.system.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set random seed for reproducibility
    torch.manual_seed(config.system.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.system.seed)

def main():
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup environment
    setup_environment(config, args)
    
    # Get data using the data factory
    data = get_data(config.data.data_type, config.system.device)
    
    # Initialize trainer for inference
    predictor = STAGEDPredictor(
        data=data,
        genes=data['genes'],
        ligand_receptor_pairs=data['ligand_receptor_pairs'],
        receptor_gene_pairs=data['receptor_gene_pairs'],
        cell_type_assignments=data['cell_type_assignments'],
        prior_grns=data['prior_grns'],
        config=config,
        checkpoint_path=args.checkpoint_path
    )
    
    # Get prediction parameters
    # Ensure we have enough history for model lags
    min_time = max(predictor.model.delta_gl, predictor.model.delta_lr, predictor.model.delta_rg, predictor.model.delta_gg)
    initial_time = args.initial_time if args.initial_time is not None else min_time
    initial_time = max(initial_time, min_time)

    prediction_steps = args.prediction_steps if args.prediction_steps is not None else 10
    
    # Run prediction
    inference_output = predictor.inference(
        initial_time=initial_time,
        prediction_steps=prediction_steps,
        store_attention=config.inference.store_attention if hasattr(config.inference, 'store_attention') else False,
        cell_ids=args.cell_type_id # choose the cell type to predict for. If None, predict for all cells.
    )

    # Save predictions
    output_path = predictor.save_predictions(
        predictions=inference_output,
        initial_time=0,
        prediction_steps=10,
        output_dir=config.inference.output_dir
    )
    
    # Print summary
    print_inference_summary(inference_output)
    print(f"\nPredictions saved to: {output_path}")
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()