import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import os
from datetime import datetime
import pickle

from src.models.staged import STAGED
from src.data.data_processor import DataProcessor
from src.config.config import Config

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
        

class STAGEDPredictor:
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        genes: List[str],
        ligand_receptor_pairs: List[tuple],
        receptor_gene_pairs: List[tuple],
        cell_type_assignments: Any,
        prior_grns: Dict[Any, Any],
        config: Config,
        checkpoint_path: Optional[str] = None,
    ):
        # Setup configuration
        self.config = config
        self.device = torch.device(config.system.device) if config.system.device != "auto" else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint if provided to get model configuration
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            if 'config' in checkpoint:
                # Use configuration from checkpoint
                model_config = checkpoint['config'].model
            else:
                print("Warning: No configuration found in checkpoint, using provided config")
                model_config = config.model
        else:
            model_config = config.model

        # Store checkpoint path for metadata
        self.checkpoint_path = checkpoint_path

        # Initialize model with proper configuration
        self.model = STAGED(
            num_genes=len(genes),
            hidden_dim=model_config.hidden_dim,
            num_gat_layers=model_config.num_gat_layers,
            num_mlp_layers=model_config.num_mlp_layers,
            dropout=model_config.dropout,
            delta_gl=model_config.delta_gl if hasattr(model_config, 'delta_gl') else 1,
            delta_lr=model_config.delta_lr if hasattr(model_config, 'delta_lr') else 5,
            delta_rg=model_config.delta_rg if hasattr(model_config, 'delta_rg') else 3,
            delta_gg=model_config.delta_gg if hasattr(model_config, 'delta_gg') else 7,
            add_self_loops=model_config.add_self_loops if hasattr(model_config, 'add_self_loops') else False,
        ).to(self.device)

        # Load model weights if checkpoint provided
        if checkpoint_path:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"Loaded model weights from {checkpoint_path}")

        # Initialize data processor
        self.data_processor = DataProcessor(
            genes=genes,
            ligand_receptor_pairs=ligand_receptor_pairs,
            receptor_gene_pairs=receptor_gene_pairs,
            cell_type_assignments=cell_type_assignments,
            prior_grns=prior_grns,
            device=self.device,
            distance_threshold=config.data.distance_threshold if hasattr(config.data, 'distance_threshold') else 1.0,
            batch_size=config.training.batch_size if hasattr(config.training, 'batch_size') else 32,
            model=self.model
        )
        
        # Process data
        self.processed_data = self.data_processor.preprocess_data(data)
        
        # Get minimum time point based on model lags
        self.min_time = max(
            self.model.delta_gl,
            self.model.delta_lr,
            self.model.delta_rg,
            self.model.delta_gg
        )

        self.ode_func = self.data_processor.ode_func

    def save_predictions(
        self,
        predictions: InferenceOutput,
        initial_time: int,
        prediction_steps: int,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Save predictions to disk with metadata.
        
        Args:
            predictions: InferenceOutput object containing predictions and metadata
            initial_time: Initial time point used for predictions
            prediction_steps: Number of prediction steps
            output_dir: Optional custom output directory. If None, uses config.system.output_dir
            
        Returns:
            Path to the saved predictions file
        """
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(self.config.system.output_dir, "predictions")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare output with metadata
        output = {
            'predictions': predictions.predictions.cpu().numpy(),
            'gene_names': predictions.genes,
            'time_points': predictions.time_points,
            'metadata': {
                'initial_time': initial_time,
                'prediction_steps': prediction_steps,
                'model_checkpoint': self.checkpoint_path,
                'prediction_time': datetime.now().isoformat(),
                'data_type': self.config.data.data_type,
                'prediction_mode': predictions.prediction_mode,
                'model_config': {
                    'hidden_dim': self.model.hidden_dim,
                    'num_gat_layers': self.model.num_gat_layers,
                    'num_mlp_layers': self.model.num_mlp_layers,
                    'dropout': self.model.dropout,
                    'delta_gl': self.model.delta_gl,
                    'delta_lr': self.model.delta_lr,
                    'delta_rg': self.model.delta_rg,
                    'delta_gg': self.model.delta_gg,
                },
                'cell_type_filter': predictions.cell_type_filter,
            }
        }
        
        # Add attention weights if available
        if predictions.attention_weights is not None:
            output['attention_weights'] = predictions.attention_weights.cpu().numpy()
        
        # Save to file using pickle
        output_path = os.path.join(output_dir, f"predictions_{initial_time}_{prediction_steps}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(output, f)
        
        print(f"Saved predictions to {output_path}")
        return output_path

    def inference(
        self,
        initial_time: int,
        prediction_steps: int,
        cell_ids: Optional[List[int]] = None,
        store_attention: bool = False
    ) -> InferenceOutput:
        """
        Generate predictions for gene expression over time.
        
        Args:
            initial_time: Starting time point for predictions
            prediction_steps: Number of time steps to predict
            cell_ids: Optional list of cell IDs to predict for. If None, predicts for all cells.
            store_attention: Whether to store attention weights during prediction
            
        Returns:
            InferenceOutput object containing:
                - predictions: Tensor of shape (prediction_steps, n_cells, n_genes)
                - attention_weights: Optional tensor of attention weights if store_attention=True
                - time_points: List of time points
                - cell_type_filter: Optional cell type filter used
                - prediction_mode: Mode used for prediction
                - model_config: Model configuration
                - genes: List of gene names
        """
        self.model.eval()
        with torch.no_grad():
            # Initialize storage
            predictions = []
            attention_weights = [] if store_attention else None
            
            # Get prediction method based on mode
            predict_method = (
                self.model.predict_ode
            )
            
            # Get method-specific kwargs
            method_kwargs = {}
            if self.config.training.prediction_mode == "ode":
                method_kwargs["method"] = (
                    self.config.training.ode_method 
                    if hasattr(self.config.training, 'ode_method') 
                    else 'rk4'
                )
            
            # Generate predictions one step at a time
            current_time = initial_time
            for _ in range(prediction_steps):
                # Get prediction for current time point
                output = predict_method(
                    data=self.processed_data,
                    time_point=current_time,
                    cell_ids=cell_ids,
                    store_attention=store_attention,
                    ode_func=self.ode_func,
                    **method_kwargs
                )
                
                # Add time dimension to predictions if needed
                pred = output.predictions
                if self.config.training.prediction_mode != "ode":
                    pred = pred.unsqueeze(0)
                
                predictions.append(pred)
                
                if store_attention and output.attention_weights is not None:
                    attention_weights.append(output.attention_weights)
                
                current_time += 1
            
            # Stack predictions along time dimension
            predictions = torch.cat(predictions, dim=0)  # Shape: (times, cells, genes)

            if store_attention and attention_weights:
                attention_weights = torch.stack(attention_weights, dim=0)
            
            # Create time points list
            time_points = list(range(initial_time, initial_time + prediction_steps))
            
            return InferenceOutput(
                predictions=predictions,
                attention_weights=attention_weights if store_attention else None,
                time_points=time_points,
                cell_type_filter=self.config.inference.cell_type_filter if hasattr(self.config.inference, 'cell_type_filter') else None,
                prediction_mode=self.config.training.prediction_mode,
                model_config=self.config.model,
                genes=self.processed_data.genes
            )