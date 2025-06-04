#!/usr/bin/env python3
"""
STAGED Model Training Main Interface

This script provides a command-line interface for training STAGED models
with different prediction modes and data types.

Usage:
    python main.py --mode train --config config/train_config.yaml
    python main.py --mode eval --config config/eval_config.yaml
    python main.py --mode predict --config config/predict_config.yaml
"""

import argparse
import torch

from src.trainer.predictor import save_predictions, InferenceOutput, print_inference_summary
from src.config.config import load_config
from src.utils.data_factory import get_data
from src.trainer.trainer import STAGEDTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='STAGED: Spatiotemporal Analysis of Gene Expression Dynamics')
    
    # Core arguments
    parser.add_argument('--mode', 
                       choices=['train', 'eval', 'inference'], 
                       required=True,
                       help='Operation mode: train model, evaluate model, or make predictions')
    
    parser.add_argument('--config', 
                       type=str, 
                       required=True,
                       help='Path to configuration file (YAML format)')
    
    # Optional arguments for overriding config
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
    
    # Prediction-specific arguments
    parser.add_argument('--initial_time',
                       type=int,
                       default=None,
                       help='Initial time point for predictions')
    
    parser.add_argument('--prediction_steps',
                       type=int,
                       default=None,
                       help='Number of time steps to predict')
    
    parser.add_argument('--checkpoint_path',
                       type=str,
                       default=None,
                       help='Path to model checkpoint. Required for inference mode.')
    
    parser.add_argument('--cell_type_id',
                       type=int,
                       default=None,
                       help='ID of the cell type to perform inference on')

    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == 'inference' and args.checkpoint_path is None:
        parser.error("--checkpoint_path is required for inference mode")
    
    return args

def setup_environment(config, args):
    """Setup the training environment."""
    # Override config with command line arguments if provided
    if args.device:
        config.system.device = args.device
    if args.output_dir:
        config.system.output_dir = args.output_dir
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
    
    # Initialize trainer (this is used for training and inference.)
    trainer = STAGEDTrainer(
        data=data,
        genes=data['genes'],
        ligand_receptor_pairs=data['ligand_receptor_pairs'],
        receptor_gene_pairs=data['receptor_gene_pairs'],
        cell_type_assignments=data['cell_type_assignments'],
        prior_grns=data['prior_grns'],
        config=config
    )

    # Execute based on mode
    if args.mode == 'train':
        trainer.fit()
        
    elif args.mode == 'inference':
        # Load the specified model checkpoint
        trainer.load_checkpoint(args.checkpoint_path)
        
        # Get prediction parameters
        initial_time = args.initial_time if args.initial_time is not None else 0
        prediction_steps = args.prediction_steps if args.prediction_steps is not None else 10
        
        # Run prediction
        predictions = trainer.inference(
            initial_time=initial_time,
            prediction_steps=prediction_steps,
            store_attention=config.inference.store_attention if hasattr(config.inference, 'store_attention') else False
        )
        
        # Create InferenceOutput object
        inference_output = InferenceOutput(
            predictions=predictions['predictions'],
            attention_weights=predictions['attention_weights'],
            time_points=predictions['time_points'].tolist(),
            cell_type_filter=args.cell_type_id,
            prediction_mode=config.training.prediction_mode,
            model_config=config.model,
            genes=config.data.genes if hasattr(config.data, 'genes') else None
        )
        
        # Save predictions
        output_path = save_predictions(
            predictions=predictions,
            config=config,
            initial_time=initial_time,
            prediction_steps=prediction_steps,
            model_path=args.checkpoint_path
        )
        
        # Print summary
        print_inference_summary(inference_output)
        
        print(f"\nPredictions saved to: {output_path}")
        
    ##TODO: Implement evaluation mode
    # elif args.mode == 'eval':
    #     trainer, _ = initialize_trainer(config, args.checkpoint_path)
    #     evaluator = STAGEDEvaluator(trainer)
    #     evaluator.evaluate()
    print(f"\n{args.mode.capitalize()} completed successfully!")

if __name__ == "__main__":
    main()