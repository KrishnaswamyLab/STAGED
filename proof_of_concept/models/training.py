import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from models.staged import STAGED
from models.inference_processor import STAGEDProcessor, PredictionOutput, ODEPredictionOutput

@dataclass
class ModelConfig:
    """Configuration for STAGED model architecture"""
    hidden_dim: int = 64
    num_gat_layers: int = 1
    num_mlp_layers: int = 2
    dropout: float = 0.1
    delta_gl: int = 1
    delta_lr: int = 5
    delta_rg: int = 3
    delta_gg: int = 7
    add_self_loops: bool = False

@dataclass
class TrainingConfig:
    """Configuration for training"""
    max_iterations: int = 1000
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    batch_size: int = 32
    distance_threshold: float = 1.0
    device: Optional[torch.device] = None
    model_config: Optional[ModelConfig] = None
    output_model_path: str = "trained_model.pth"

@dataclass
class TrainingOutput:
    """Training results"""
    loss_history: List[float]
    model: STAGED

def train_staged_model(
    data: Dict[str, torch.Tensor],
    genes: List[str],
    ligand_receptor_pairs: List[tuple],
    receptor_gene_pairs: List[tuple],
    cell_type_assignments: Any,
    prior_grns: Dict[Any, Any],
    prediction_mode: str = "one_step",
    k_steps: Optional[int] = None,
    ode_eval_times: Optional[torch.Tensor] = None,
    ode_method: str = 'rk4',
    config: Optional[TrainingConfig] = None,
) -> TrainingOutput:
    """
    Train a STAGED model using the specified prediction mode.
    
    Args:
        data: Dictionary containing:
            - gene_expression: Tensor of shape (n_time_points, n_cells, n_genes)
            - cell_positions: Tensor of shape (n_time_points, n_cells, 2)
            - n_cells: Number of cells
        genes: List of gene identifiers
        ligand_receptor_pairs: List of (ligand, receptor) pairs
        receptor_gene_pairs: List of (receptor, gene) pairs
        cell_type_assignments: Cell type assignments
        prior_grns: Dictionary of prior GRNs
        prediction_mode: One of ["one_step", "k_step", "full", "ode"]
        k_steps: Number of steps for k-step prediction (required if mode is "k_step")
        ode_eval_times: Times to evaluate ODE at (required if mode is "ode")
        ode_method: ODE integration method for ODE mode
        config: Training configuration
        
    Returns:
        TrainingOutput containing trained model and loss history
    """
    # Setup configuration
    if config is None:
        config = TrainingConfig()
    if config.model_config is None:
        config.model_config = ModelConfig()
    device = config.device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Validate ODE mode parameters
    if prediction_mode == "ode":
        if ode_eval_times is None:
            raise ValueError("ode_eval_times must be provided for ODE prediction mode")
        ode_eval_times = ode_eval_times.to(device)
    
    # Initialize model
    model = STAGED(
        num_genes=len(genes),
        hidden_dim=config.model_config.hidden_dim,
        num_gat_layers=config.model_config.num_gat_layers,
        num_mlp_layers=config.model_config.num_mlp_layers,
        dropout=config.model_config.dropout,
        delta_gl=config.model_config.delta_gl,
        delta_lr=config.model_config.delta_lr,
        delta_rg=config.model_config.delta_rg,
        delta_gg=config.model_config.delta_gg,
        add_self_loops=config.model_config.add_self_loops,
    ).to(device)
    
    # Initialize processor
    processor = STAGEDProcessor(
        model=model,
        genes=genes,
        ligand_receptor_pairs=ligand_receptor_pairs,
        receptor_gene_pairs=receptor_gene_pairs,
        cell_type_assignments=cell_type_assignments,
        prior_grns=prior_grns,
        batch_size=config.batch_size,
        distance_threshold=config.distance_threshold,
        device=device
    )
    
    # Setup ODE if needed
    if prediction_mode == "ode":
        processor.setup_ode(data['gene_expression'])
    if prediction_mode == "ode_new":
        processor.setup_ode(data['gene_expression'])
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Initialize loss function
    criterion = nn.MSELoss()
    
    # Get minimum time point based on model lags
    min_time = max(model.delta_gl, model.delta_lr, model.delta_rg, model.delta_gg)
    
    # Validate k_steps if in k_step mode
    if prediction_mode == "k_step":
        if k_steps is None:
            raise ValueError("k_steps must be provided for k_step prediction mode")
        if k_steps >= data['gene_expression'].shape[0] - min_time:
            raise ValueError(
                f"k_steps ({k_steps}) must be less than available prediction steps "
                f"({data['gene_expression'].shape[0] - min_time})"
            )
    
    # Training loop
    loss_history = []
    pbar = tqdm(range(config.max_iterations), desc="Training")
    
    for iteration in pbar:
        optimizer.zero_grad()
        
        if prediction_mode == "ode":
            # ODE training mode
            total_loss = 0.0
            n_training_segments = 0
            
            # Train on segments of the time series
            for start_time in range(min_time, data['gene_expression'].shape[0] - len(ode_eval_times) + 1):
                # Set up time range for this segment
                initial_time = float(start_time)
                eval_times_segment = ode_eval_times + start_time
                
                # Get ODE predictions
                ode_output = processor.predict_ode(
                    data=data,
                    initial_time=initial_time,
                    eval_times=eval_times_segment,
                    method=ode_method,
                    store_attention=False  # Don't store attention during training for efficiency
                )
                
                # Get ground truth targets for this segment
                target_indices = [int(t) for t in eval_times_segment if t < data['gene_expression'].shape[0]]
                if len(target_indices) == len(eval_times_segment):
                    target = data['gene_expression'][target_indices].to(device)
                    
                    # Compute loss for this segment
                    segment_loss = criterion(ode_output.predictions[:len(target_indices)], target)
                    total_loss += segment_loss
                    n_training_segments += 1
            
            if n_training_segments == 0:
                raise ValueError("No valid training segments found for ODE mode")
            
            # Average loss across segments
            loss = total_loss / n_training_segments
        
        elif prediction_mode == "ode_new":
            # ODE training mode - individual timepoint predictions (like original structure)
            # Initialize prediction collection tensor
            n_prediction_steps = data['gene_expression'].shape[0] - min_time
            predictions = torch.zeros(
                (n_prediction_steps, data['n_cells'], len(genes)),
                device=device
            )
            
            # Generate ODE predictions for each time point individually
            
            for t in range(min_time, data['gene_expression'].shape[0]):
                # Set up single timepoint evaluation
                initial_time = float(t)
                eval_times_single = torch.tensor([t], dtype=torch.float)  # Single timepoint
                
                # Get ODE predictions for this single timepoint
                ode_output = processor.predict_ode_new(
                    data=data,
                    time_point=t,
                    method=ode_method,
                    store_attention=False  # Don't store attention during training for efficiency
                )
                
                # Store prediction for current time point
                predictions[t - min_time] = ode_output.predictions[0]  # Take first (and only) prediction
            
            # Compute loss (same as original approach)
            target = data['gene_expression'][min_time:].to(device)
            loss = criterion(predictions, target)
        else:
            # Original training modes (one_step, k_step, full)
            # Initialize prediction collection tensor
            n_prediction_steps = data['gene_expression'].shape[0] - min_time
            predictions = torch.zeros(
                (n_prediction_steps, data['n_cells'], len(genes)),
                device=device
            )
            
            # Generate predictions for each time point
            for t in range(min_time, data['gene_expression'].shape[0]):
                # Get predictions for current time point
                output = processor.predict(
                    data=data,
                    time_point=t
                )
                
                if prediction_mode == "one_step":
                    # Store one-step prediction
                    predictions[t - min_time] = output.predictions
                else:
                    raise NotImplementedError(
                        f"Prediction mode {prediction_mode} not yet implemented"
                    )
            
            # Compute loss
            target = data['gene_expression'][min_time:].to(device)
            loss = criterion(predictions, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Record loss
        loss_history.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    torch.save(model.state_dict(), config.output_model_path)
    return TrainingOutput(
        loss_history=loss_history,
        model=model
    ) 