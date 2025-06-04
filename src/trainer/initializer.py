import os
import torch
from src.trainer.trainer import STAGEDTrainer
from src.utils.data_factory import get_data

def initialize_trainer(config):
    """
    Initialize the STAGED trainer.
    
    Args:
        config: Configuration object
        checkpoint_path: Optional path to load a checkpoint for resuming training
        
    Returns:
        trainer: Initialized STAGEDTrainer instance
        best_model_path: Path to the checkpoint if loaded, None otherwise
    """

    return trainer