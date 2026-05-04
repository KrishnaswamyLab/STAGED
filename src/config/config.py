# config/config.py
import yaml
import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    """Data configuration settings"""
    data_dir: str = "data"
    expression_data: str = "data/gene_expression.csv"
    positions_data: str = "data/cell_positions.csv"
    lr_pairs_data: str = "data/ligand_receptor_pairs.csv"
    cell_types_data: str = "data/cell_types.csv"
    prior_grns_data: str = "data/prior_grns.csv"
    data_type: str = "oscillatory"  # or "hex_grid", "damped_oscillator"
    sim_file: str = "100_simulation_results.pkl"
    distance_threshold: float = 10.0
    validation_fraction: float = 0.2
    train_end_time: Optional[float] = None

@dataclass
class ModelConfig:
    """Model configuration settings"""
    hidden_dim: int = 64
    num_gat_layers: int = 1
    num_mlp_layers: int = 3
    dropout: float = 0.1
    delta_gl: int = 1  # gene -> ligand
    delta_lr: int = 2  # ligand -> receptor
    delta_rg: int = 1  # receptor -> gene
    delta_gg: int = 0  # gene -> gene (should be 0 for ODE mode)
    add_self_loops: bool = True

@dataclass
class TrainingConfig:
    """Training configuration settings"""
    prediction_mode: str = "one_step"  # "one_step", "k_step", "ode"
    max_iterations: int = 50
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 0.01
    weight_decay: float = 1e-5
    patience: int = 10
    k_steps: int = 3  # for k_step mode
    ode_method: str = "rk4"  # for ODE modes
    time_points_per_iter: Optional[int] = None

@dataclass
class SystemConfig:
    """System configuration settings"""
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    seed: int = 42
    output_dir: str = "results"
    visualize: bool = True

@dataclass
class LoggingConfig:
    """Logging configuration settings"""
    level: str = "INFO"
    save_logs: bool = True

@dataclass
class InferenceConfig:
    """Inference configuration settings"""
    store_attention: bool = True
    output_dir: str = "results"

@dataclass
class Config:
    """Single consolidated configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
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
    
    # Create nested config objects
    data_config = DataConfig(**(yaml_config.get('data', {})))
    model_config = ModelConfig(**(yaml_config.get('model', {})))
    training_config = TrainingConfig(**(yaml_config.get('training', {})))
    system_config = SystemConfig(**(yaml_config.get('system', {})))
    logging_config = LoggingConfig(**(yaml_config.get('logging', {})))
    inference_config = InferenceConfig(**(yaml_config.get('inference', {})))
    
    # Create main config object
    config = Config(
        data=data_config,
        model=model_config,
        training=training_config,
        system=system_config,
        logging=logging_config,
        inference=inference_config
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(config.system.output_dir, exist_ok=True)
    
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
        'data': {
            'data_dir' : config.data.data_dir,
            'expression_data': config.data.expression_data,
            'positions_data': config.data.positions_data,
            'lr_pairs_data': config.data.lr_pairs_data,
            'cell_types_data': config.data.cell_types_data,
            'prior_grns_data': config.data.prior_grns_data,
            'data_type': config.data.data_type,
            'distance_threshold': config.data.distance_threshold,
            'validation_fraction': config.data.validation_fraction,
            'train_end_time': config.data.train_end_time,
        },
        'model': {
            'hidden_dim': config.model.hidden_dim,
            'num_gat_layers': config.model.num_gat_layers,
            'num_mlp_layers': config.model.num_mlp_layers,
            'dropout': config.model.dropout,
            'delta_gl': config.model.delta_gl,
            'delta_lr': config.model.delta_lr,
            'delta_rg': config.model.delta_rg,
            'delta_gg': config.model.delta_gg,
            'add_self_loops': config.model.add_self_loops,
        },
        'training': {
            'prediction_mode': config.training.prediction_mode,
            'max_iterations': config.training.max_iterations,
            'num_epochs': config.training.num_epochs,
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay,
            'patience': config.training.patience,
            'k_steps': config.training.k_steps,
            'ode_method': config.training.ode_method,
        },
        'system': {
            'device': config.system.device,
            'seed': config.system.seed,
            'output_dir': config.system.output_dir,
            'visualize': config.system.visualize,
        },
        'logging': {
            'level': config.logging.level,
            'save_logs': config.logging.save_logs,
        },
        'inference': {
            'store_attention': config.inference.store_attention,
            'output_dir': config.inference.output_dir,
        }
    }
    
    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)