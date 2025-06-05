import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import os
import json
from datetime import datetime

from src.models.staged import STAGED
from src.data.data_processor import DataProcessor
from src.config.config import Config

@dataclass
class TrainingOutput:
    """Training results"""
    loss_history: List[float]
    model: STAGED
    best_model_path: str
    checkpoint_dir: str


@dataclass
class PredictionOutput:
    """Structured output from prediction"""
    predictions: torch.Tensor
    attention_weights: Tuple[torch.Tensor, torch.Tensor]  # (edges, values)
    node_pointers: torch.Tensor

class STAGEDTrainer:
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        genes: List[str],
        ligand_receptor_pairs: List[tuple],
        receptor_gene_pairs: List[tuple],
        cell_type_assignments: Any,
        prior_grns: Dict[Any, Any],
        config: Config,

    ):
        # Setup configuration
        self.config = config
        self.device = torch.device(config.system.device) if config.system.device != "auto" else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create checkpoint directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = os.path.join(config.system.output_dir, "checkpoints", f"checkpoints_{timestamp}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        

        # Initialize model
        self.model = STAGED(
            num_genes=len(genes),
            hidden_dim=config.model.hidden_dim,
            num_gat_layers=config.model.num_gat_layers,
            num_mlp_layers=config.model.num_mlp_layers,
            dropout=config.model.dropout,
            delta_gl=config.model.delta_gl if hasattr(config.model, 'delta_gl') else 1,
            delta_lr=config.model.delta_lr if hasattr(config.model, 'delta_lr') else 5,
            delta_rg=config.model.delta_rg if hasattr(config.model, 'delta_rg') else 3,
            delta_gg=config.model.delta_gg if hasattr(config.model, 'delta_gg') else 7,
            add_self_loops=config.model.add_self_loops if hasattr(config.model, 'add_self_loops') else False,
        ).to(self.device)

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
        
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Get minimum time point based on model lags
        self.min_time = max(
            self.model.delta_gl,
            self.model.delta_lr,
            self.model.delta_rg,
            self.model.delta_gg
        )
        
        # Initialize checkpoint tracking
        self.best_loss = float('inf')
        self.best_model_path = None
        
        # Save initial config
        self._save_config()

    def _save_config(self):
        """Save the configuration to the checkpoint directory"""
        config_path = os.path.join(self.checkpoint_dir, 'config.json')
        config_dict = {
            'data': {
                'n_genes': len(self.data_processor.genes),
                'n_cells': self.processed_data.n_cells,
                'n_time_points': self.processed_data.n_time_points,
            },
            'model': {
                'hidden_dim': self.config.model.hidden_dim,
                'num_gat_layers': self.config.model.num_gat_layers,
                'num_mlp_layers': self.config.model.num_mlp_layers,
                'dropout': self.config.model.dropout,
                'delta_gl': self.config.model.delta_gl if hasattr(self.config.model, 'delta_gl') else 1,
                'delta_lr': self.config.model.delta_lr if hasattr(self.config.model, 'delta_lr') else 5,
                'delta_rg': self.config.model.delta_rg if hasattr(self.config.model, 'delta_rg') else 3,
                'delta_gg': self.config.model.delta_gg if hasattr(self.config.model, 'delta_gg') else 7,
                'add_self_loops': self.config.model.add_self_loops if hasattr(self.config.model, 'add_self_loops') else False,
            },
            'training': {
                'prediction_mode': self.config.training.prediction_mode,
                'max_iterations': self.config.training.max_iterations if hasattr(self.config.training, 'max_iterations') else 1000,
                'batch_size': self.config.training.batch_size,
                'learning_rate': self.config.training.learning_rate,
                'weight_decay': self.config.training.weight_decay,
            },
            'system': {
                'device': str(self.device),
                'checkpoint_dir': self.checkpoint_dir,
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def _save_best_checkpoint(self, iteration: int, loss: float):
        """Save the best model checkpoint"""
        if loss < self.best_loss:
            checkpoint = {
                'iteration': iteration,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'config': self.config,
            }
            
            best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            self.best_model_path = best_model_path
            self.best_loss = loss

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a model checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Set model to eval mode
        self.model.eval()
        
        print(f"Loaded checkpoint from {checkpoint_path}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        if self.config.training.prediction_mode == "ode":
            total_loss = self._train_ode_mode()
        else:
            total_loss = self._train_standard_mode()
            
        return total_loss

    def _train_ode_mode(self):
        """Train using ODE mode"""
        # Initialize prediction collection tensor
        n_prediction_steps = self.processed_data.gene_expression.shape[0] - self.min_time
        predictions = torch.zeros(
            (n_prediction_steps, self.processed_data.n_cells, len(self.data_processor.genes)),
            device=self.device
        )
        
        # Generate ODE predictions for each time point individually
        for t in range(self.min_time, self.processed_data.gene_expression.shape[0]):
            # Get ODE predictions for this single timepoint
            output = self.model.predict_ode(
                data=self.processed_data,
                time_point=t,
                method=self.config.training.ode_method if hasattr(self.config.training, 'ode_method') else 'rk4',
                store_attention=False,
                ode_func=self.data_processor._create_ode_function()
            )
            
            # Store prediction for current time point
            predictions[t - self.min_time] = output.predictions[0]
        
        # Compute loss
        target = self.processed_data.gene_expression[self.min_time:].to(self.device)
        return self.criterion(predictions, target)

    def _train_standard_mode(self):
        """Train using standard modes (one_step, k_step)"""
        # Initialize prediction collection tensor
        n_prediction_steps = self.processed_data.gene_expression.shape[0] - self.min_time
        predictions = torch.zeros(
            (n_prediction_steps, self.processed_data.n_cells, len(self.data_processor.genes)),
            device=self.device
        )

        # Generate predictions for each time point
        for t in range(self.min_time, self.processed_data.gene_expression.shape[0]):
            # Get predictions for current time point
            output = self.model.predict(
                data=self.processed_data,
                time_point=t
            )
            
            if self.config.training.prediction_mode == "one_step":
                # Store one-step prediction
                predictions[t - self.min_time] = output.predictions
            else:
                raise NotImplementedError(
                    f"Prediction mode {self.config.training.prediction_mode} not yet implemented"
                )
        
        # Compute loss
        target = self.processed_data.gene_expression[self.min_time:].to(self.device)
        return self.criterion(predictions, target)

    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        with torch.no_grad():
            return self.train_epoch()

    def fit(self):
        """Train the model"""
        loss_history = []
        pbar = tqdm(range(self.config.training.max_iterations if hasattr(self.config.training, 'max_iterations') else 1000), desc="Training")
        
        for iteration in pbar:
            self.optimizer.zero_grad()
            loss = self.train_epoch()
            loss.backward()
            self.optimizer.step()
            
            # Record loss
            loss_history.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            # Save best model if this is the best so far
            self._save_best_checkpoint(iteration + 1, loss.item())
        
        # Save training metadata
        metadata = {
            'loss_history': loss_history,
            'best_loss': self.best_loss,
            'best_model_path': self.best_model_path,
            'final_iteration': len(loss_history),
            'training_time': datetime.now().isoformat(),
        }
        metadata_path = os.path.join(self.checkpoint_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return TrainingOutput(
            loss_history=loss_history,
            model=self.model,
            best_model_path=self.best_model_path,
            checkpoint_dir=self.checkpoint_dir
        )

    def inference(
        self,
        initial_time: int,
        prediction_steps: int,
        cell_ids: Optional[List[int]] = None,
        store_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate predictions for gene expression over time.
        
        Args:
            initial_time: Starting time point for predictions
            prediction_steps: Number of time steps to predict
            cell_ids: Optional list of cell IDs to predict for. If None, predicts for all cells.
            store_attention: Whether to store attention weights during prediction
            
        Returns:
            Dictionary containing:
                - predictions: Tensor of shape (prediction_steps, n_cells, n_genes)
                - time_points: Tensor of shape (prediction_steps,)
                - attention_weights: Optional tensor of attention weights if store_attention=True
        """
        self.model.eval()
        with torch.no_grad():
            # Initialize storage
            predictions = []
            attention_weights = [] if store_attention else None
            
            # Get prediction method based on mode
            predict_method = (
                self.model.predict_ode 
                if self.config.training.prediction_mode == "ode" 
                else self.model.predict
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
                    ode_func=self.data_processor._create_ode_function(),
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
            
            return {
                'predictions': predictions,
                'gene_names': self.data_processor.genes,
                'time_points': torch.arange(initial_time, initial_time + prediction_steps, device=self.device),
                'attention_weights': attention_weights if store_attention else None
            }