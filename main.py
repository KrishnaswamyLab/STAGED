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
import sys
from src.training.trainer import STAGEDTrainer
from src.evaluation.evaluator import STAGEDEvaluator
from src.prediction.predictor import STAGEDPredictor
from config.config import load_config

def main():
    parser = argparse.ArgumentParser(description='STAGED: Spatiotemporal Analysis of Gene Expression Dynamics')
    
    # Core arguments. The rest is on the config file.
    parser.add_argument('--mode', 
                       choices=['train', 'eval', 'predict'], 
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
    
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments if provided
    if args.device:
        config.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed:
        config.seed = args.seed
    
    # Execute based on mode
    if args.mode == 'train':
        trainer = STAGEDTrainer(config)
        trainer.train()
        
    elif args.mode == 'eval':
        evaluator = STAGEDEvaluator(config)
        evaluator.evaluate()
        
    elif args.mode == 'predict':
        predictor = STAGEDPredictor(config)
        predictor.predict()
        
    print(f"\n{args.mode.capitalize()} completed successfully!")

if __name__ == "__main__":
    main()