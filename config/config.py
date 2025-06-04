# config/config.py
import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Config:
    """Single consolidated configuration class"""
    # Experiment settings
    experiment_name: str = "default_experiment"
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    log_dir: str = "./results/logs"
    checkpoint_dir: str = "./results/checkpoints"
    save_every: int = 10
    
    # Model configuration
    model_name: str = "GCN"
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    activation: str = "relu"
    normalization: str = "batch_norm"
    
    # Optimal Transport configuration
    ot_method: str = "sinkhorn"  # sinkhorn, emd, gromov_wasserstein
    ot_reg: float = 0.1  # regularization parameter
    ot_max_iter: int = 1000
    ot_tol: float = 1e-6
    ot_cost_function: str = "euclidean"  # euclidean, cosine, hamming
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 0.01
    num_epochs: int = 100
    optimizer: str = "adam"
    scheduler: str = "step"
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    
    # Data configuration
    dataset_name: str = "Cora"
    data_dir: str = "./data"
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True
    
    # Loss function weights (if using multiple losses)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "reconstruction": 1.0,
        "ot_regularization": 0.1,
        "classification": 1.0
    })
    
    # Evaluation metrics to track
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "f1_score", "wasserstein_distance"
    ])
    
    # Logging configuration
    log_level: str = "INFO"
    log_frequency: int = 10  # log every N steps
    save_predictions: bool = False
    
    # Reproducibility settings
    deterministic: bool = True
    benchmark: bool = False  # cudnn benchmark

def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file and merge with defaults
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded parameters
    """
    # Load YAML file
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    if yaml_config is None:
        yaml_config = {}
    
    # Create config object from YAML data
    # This will use defaults for any missing keys
    config = Config(**yaml_config)
    
    # Create directories if they don't exist
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    return config

def save_config(config: Config, save_path: str) -> None:
    """
    Save configuration to YAML file
    
    Args:
        config: Config object to save
        save_path: Path where to save the configuration
    """
    # Convert config to dictionary
    config_dict = {
        # Experiment settings
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        'device': config.device,
        'log_dir': config.log_dir,
        'checkpoint_dir': config.checkpoint_dir,
        'save_every': config.save_every,
        
        # Model configuration
        'model_name': config.model_name,
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
        'activation': config.activation,
        'normalization': config.normalization,
        
        # Optimal Transport configuration
        'ot_method': config.ot_method,
        'ot_reg': config.ot_reg,
        'ot_max_iter': config.ot_max_iter,
        'ot_tol': config.ot_tol,
        'ot_cost_function': config.ot_cost_function,
        
        # Training configuration
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'num_epochs': config.num_epochs,
        'optimizer': config.optimizer,
        'scheduler': config.scheduler,
        'weight_decay': config.weight_decay,
        'early_stopping_patience': config.early_stopping_patience,
        'gradient_clip_norm': config.gradient_clip_norm,
        
        # Data configuration
        'dataset_name': config.dataset_name,
        'data_dir': config.data_dir,
        'train_split': config.train_split,
        'val_split': config.val_split,
        'test_split': config.test_split,
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
        
        # Additional settings
        'loss_weights': config.loss_weights,
        'metrics': config.metrics,
        'log_level': config.log_level,
        'log_frequency': config.log_frequency,
        'save_predictions': config.save_predictions,
        'deterministic': config.deterministic,
        'benchmark': config.benchmark,
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)