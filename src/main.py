#!/usr/bin/env python3
"""
STAGED Model Training Main Interface

This script provides a command-line interface for training STAGED models
with different prediction modes and data types.

Usage:
    python main.py --mode train --config config/train_config.yaml
    python main.py --mode eval --config config/eval_config.yaml
"""

import argparse
import torch

from config.config import load_config
from utils.data_factory import get_data
from trainer.trainer import STAGEDTrainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='STAGED: Spatiotemporal Analysis of Gene Expression Dynamics')
    
    # Core arguments
    parser.add_argument('--mode', 
                       choices=['train', 'eval'], 
                       required=True,
                       help='Operation mode: train model or evaluate model')
    
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

    args = parser.parse_args()
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
    
    # Initialize trainer
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
        training_output = trainer.fit()
        print(f"\nBest model saved to: {training_output.best_model_path}")

    ##TODO: Implement evaluation mode
    # elif args.mode == 'eval':
    #     trainer, _ = initialize_trainer(config, args.checkpoint_path)
    #     evaluator = STAGEDEvaluator(trainer)
    #     evaluator.evaluate()
    print(f"\n{args.mode.capitalize()} completed successfully!")

if __name__ == "__main__":
    main()