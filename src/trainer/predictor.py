import os
import json
import pickle
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import torch
import glob

@dataclass
class InferenceOutput:
    """Output from STAGED model inference"""
    predictions: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    time_points: List[int] = None
    cell_type_filter: Optional[int] = None
    prediction_mode: str = None
    model_config: Optional[Any] = None
    genes: List[str] = None

def print_inference_summary(predictions: InferenceOutput):
    """Print summary of inference results"""
    print(f"\n=== Inference Summary ===")
    print(f"Prediction Mode: {predictions.prediction_mode}")
    print(f"Cell Type Filter: {predictions.cell_type_filter}")
    print(f"Time Range: {predictions.time_points[0]} - {predictions.time_points[-1]}")
    print(f"Number of time points: {len(predictions.time_points)}")
    print(f"Number of cells: {predictions.predictions.shape[1]}")
    print(f"Number of genes: {predictions.predictions.shape[2]}")
    
    # Expression statistics
    mean_expr = torch.mean(predictions.predictions).item()
    std_expr = torch.std(predictions.predictions).item()
    min_expr = torch.min(predictions.predictions).item()
    max_expr = torch.max(predictions.predictions).item()
    
    print(f"\nExpression Statistics:")
    print(f"Mean expression: {mean_expr:.4f}")
    print(f"Std expression: {std_expr:.4f}")
    print(f"Expression range: [{min_expr:.4f}, {max_expr:.4f}]")
    
    if predictions.attention_weights is not None:
        print(f"Attention weights stored: {predictions.attention_weights.shape}")
        
def save_predictions(predictions, config, initial_time, prediction_steps, model_path):
    """
    Save predictions to disk with metadata.
    
    Args:
        predictions: Dictionary containing predictions, time points, and attention weights
        config: Configuration object
        initial_time: Initial time point used for predictions
        prediction_steps: Number of prediction steps
        model_path: Path to the model checkpoint used
    """
    # Create output directory
    output_dir = os.path.join(config.system.output_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare output with metadata
    output = {
        'predictions': predictions['predictions'].cpu().numpy(),
        'gene_names': predictions['gene_names'],
        'time_points': predictions['time_points'].cpu().numpy(),
        'metadata': {
            'initial_time': initial_time,
            'prediction_steps': prediction_steps,
            'model_checkpoint': model_path,
            'prediction_time': datetime.now().isoformat(),
            'data_type': config.data.data_type,
            'prediction_mode': config.training.prediction_mode,
            'model_config': {
                'hidden_dim': config.model.hidden_dim,
                'num_gat_layers': config.model.num_gat_layers,
                'num_mlp_layers': config.model.num_mlp_layers,
                'dropout': config.model.dropout,
                'delta_gl': config.model.delta_gl if hasattr(config.model, 'delta_gl') else 1,
                'delta_lr': config.model.delta_lr if hasattr(config.model, 'delta_lr') else 5,
                'delta_rg': config.model.delta_rg if hasattr(config.model, 'delta_rg') else 3,
                'delta_gg': config.model.delta_gg if hasattr(config.model, 'delta_gg') else 7,
            },
            'cell_type_filter': config.inference.cell_type_filter if hasattr(config.inference, 'cell_type_filter') else None,
        }
    }
    
    # Add attention weights if available
    if predictions['attention_weights'] is not None:
        output['attention_weights'] = predictions['attention_weights'].cpu().numpy()
    
    # Save to file using pickle
    output_path = os.path.join(output_dir, f"predictions_{initial_time}_{prediction_steps}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)
    
    return output_path
