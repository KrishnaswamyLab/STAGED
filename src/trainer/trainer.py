import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import os
import json
from datetime import datetime

from src.models.staged import STAGED
from src.models.inference_processor import STAGEDProcessor, PredictionOutput, ODEPredictionOutput
from src.config.config import Config, load_config

@dataclass
class TrainingOutput:
    """Training results"""
    loss_history: List[float]
    model: STAGED
    best_model_path: str
    checkpoint_dir: str

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
        
        # Create variables for convenience
        self.genes = genes
        self.n_genes = len(genes)
        self.n_cells = data['n_cells']
        self.n_time_points = data['n_time_points']
        
        # Save initial config
        self._save_config()

        # Initialize model
        self.model = STAGED(
            num_genes=self.n_genes,
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
        
        # Initialize processor
        self.processor = STAGEDProcessor(
            model=self.model,
            genes=genes,
            ligand_receptor_pairs=ligand_receptor_pairs,
            receptor_gene_pairs=receptor_gene_pairs,
            cell_type_assignments=cell_type_assignments,
            prior_grns=prior_grns,
            batch_size=config.training.batch_size,
            distance_threshold=config.data.distance_threshold if hasattr(config.data, 'distance_threshold') else 1.0,
            device=self.device
        )
        
        # Setup ODE if needed
        if config.training.prediction_mode in ["ode", "ode_new"]:
            self.processor.setup_ode(data['gene_expression'])
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Initialize loss function
        self.criterion = nn.MSELoss()
        
        # Store data
        self.data = data
        
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
        
        # Validate parameters
        self._validate_parameters()

    def _save_config(self):
        """Save the configuration to the checkpoint directory"""
        config_path = os.path.join(self.checkpoint_dir, 'config.json')
        config_dict = {
            'data': {
                'n_genes': self.n_genes,
                'n_cells': self.n_cells,
                'n_time_points': self.n_time_points,
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

    def _validate_parameters(self):
        """Validate training parameters based on prediction mode"""
        if self.config.training.prediction_mode == "ode":
            if not hasattr(self.config.training, 'ode_eval_times') or self.config.training.ode_eval_times is None:
                raise ValueError("ode_eval_times must be provided for ODE prediction mode")
            self.config.training.ode_eval_times = torch.tensor(self.config.training.ode_eval_times, device=self.device)
        
        if self.config.training.prediction_mode == "k_step":
            if not hasattr(self.config.training, 'k_steps') or self.config.training.k_steps is None:
                raise ValueError("k_steps must be provided for k_step prediction mode")
            if self.config.training.k_steps >= self.data['gene_expression'].shape[0] - self.min_time:
                raise ValueError(
                    f"k_steps ({self.config.training.k_steps}) must be less than available prediction steps "
                    f"({self.data['gene_expression'].shape[0] - self.min_time})"
                )

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
        elif self.config.training.prediction_mode == "ode_new":
            total_loss = self._train_ode_new_mode()
        else:
            total_loss = self._train_standard_mode()
            
        return total_loss

    def _train_ode_mode(self):
        """Train using ODE mode"""
        total_loss = 0.0
        n_training_segments = 0
        
        # Train on segments of the time series
        for start_time in range(self.min_time, self.data['gene_expression'].shape[0] - len(self.config.ode_eval_times) + 1):
            # Set up time range for this segment
            initial_time = float(start_time)
            eval_times_segment = self.config.ode_eval_times + start_time
            
            # Get ODE predictions
            ode_output = self.processor.predict_ode(
                data=self.data,
                initial_time=initial_time,
                eval_times=eval_times_segment,
                method=self.config.training.ode_method if hasattr(self.config.training, 'ode_method') else 'rk4',
                store_attention=False
            )
            
            # Get ground truth targets for this segment
            target_indices = [int(t) for t in eval_times_segment if t < self.data['gene_expression'].shape[0]]
            if len(target_indices) == len(eval_times_segment):
                target = self.data['gene_expression'][target_indices].to(self.device)
                
                # Compute loss for this segment
                segment_loss = self.criterion(ode_output.predictions[:len(target_indices)], target)
                total_loss += segment_loss
                n_training_segments += 1
        
        if n_training_segments == 0:
            raise ValueError("No valid training segments found for ODE mode")
        
        return total_loss / n_training_segments

    def _train_ode_new_mode(self):
        """Train using ODE new mode"""
        # Initialize prediction collection tensor
        n_prediction_steps = self.data['gene_expression'].shape[0] - self.min_time
        predictions = torch.zeros(
            (n_prediction_steps, self.n_cells, self.n_genes),
            device=self.device
        )
        
        # Generate ODE predictions for each time point individually
        for t in range(self.min_time, self.data['gene_expression'].shape[0]):
            # Get ODE predictions for this single timepoint
            ode_output = self.processor.predict_ode_new(
                data=self.data,
                time_point=t,
                method=self.config.training.ode_method if hasattr(self.config.training, 'ode_method') else 'rk4',
                store_attention=False
            )
            
            # Store prediction for current time point
            predictions[t - self.min_time] = ode_output.predictions[0]
        
        # Compute loss
        target = self.data['gene_expression'][self.min_time:].to(self.device)
        return self.criterion(predictions, target)

    def _train_standard_mode(self):
        """Train using standard modes (one_step, k_step, full)"""
        # Initialize prediction collection tensor
        n_prediction_steps = self.data['gene_expression'].shape[0] - self.min_time
        predictions = torch.zeros(
            (n_prediction_steps, self.n_cells, self.n_genes),
            device=self.device
        )

        # Generate predictions for each time point
        for t in range(self.min_time, self.data['gene_expression'].shape[0]):
            # Get predictions for current time point
            output = self.processor.predict(
                data=self.data,
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
        target = self.data['gene_expression'][self.min_time:].to(self.device)
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
            if self.config.training.prediction_mode == "ode":
                # For ODE mode, we need to create evaluation times
                eval_times = torch.arange(
                    initial_time,
                    initial_time + prediction_steps,
                    device=self.device
                ).float()
                
                # Get ODE predictions
                ode_output = self.processor.predict_ode(
                    data=self.data,
                    initial_time=float(initial_time),
                    eval_times=eval_times,
                    method=self.config.training.ode_method if hasattr(self.config.training, 'ode_method') else 'rk4',
                    cell_ids=cell_ids,
                    store_attention=store_attention
                )
                
                return {
                    'predictions': ode_output.predictions,
                    'gene_names': self.genes,
                    'time_points': ode_output.eval_times,
                    'attention_weights': ode_output.attention_weights if store_attention else None
                }
                
            elif self.config.training.prediction_mode == "ode_new":
                # For ODE new mode, we predict one step at a time
                predictions = []
                attention_weights = [] if store_attention else None
                
                current_time = initial_time
                for _ in range(prediction_steps):
                    # Get prediction for current time point
                    output = self.processor.predict_ode_new(
                        data=self.data,
                        time_point=current_time,
                        method=self.config.training.ode_method if hasattr(self.config.training, 'ode_method') else 'rk4',
                        cell_ids=cell_ids,
                        store_attention=store_attention
                    )
                    
                    predictions.append(output.predictions)
                    if store_attention and output.attention_weights is not None:
                        attention_weights.append(output.attention_weights)
                    
                    current_time += 1
                
                # Stack predictions
                predictions = torch.cat(predictions, dim=0)
                if store_attention and attention_weights:
                    attention_weights = torch.stack(attention_weights, dim=0)
                
                return {
                    'predictions': predictions,
                    'gene_names': self.genes,
                    'time_points': torch.arange(initial_time, initial_time + prediction_steps, device=self.device),
                    'attention_weights': attention_weights if store_attention else None
                }
                
            else:
                # For standard modes, we predict one step at a time
                predictions = []
                attention_weights = [] if store_attention else None
                
                current_time = initial_time
                for _ in range(prediction_steps):
                    # Get prediction for current time point
                    output = self.processor.predict(
                        data=self.data,
                        time_point=current_time
                    )
                    
                    # Add time dimension to predictions (n_cells, n_genes) -> (1, n_cells, n_genes)
                    predictions.append(output.predictions.unsqueeze(0))
                    
                    if store_attention and output.attention_weights is not None:
                        attention_weights.append(output.attention_weights)
                    
                    current_time += 1
                
                # Stack predictions along time dimension
                predictions = torch.cat(predictions, dim=0)  # This will give (times, cells, genes)
                if store_attention and attention_weights:
                    attention_weights = torch.stack(attention_weights, dim=0)
                
                return {
                    'predictions': predictions,
                    'gene_names': self.genes,
                    'time_points': torch.arange(initial_time, initial_time + prediction_steps, device=self.device),
                    'attention_weights': attention_weights if store_attention else None
                }
