import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from src.data.data_processor import ProcessedData
from src.utils.ode import odeint_fixed

@dataclass
class PredictionOutput:
    """Container for model predictions"""
    predictions: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    node_pointers: Optional[torch.Tensor] = None

class STAGED(nn.Module):
    """
    STAGED (Spatiotemporal Analysis of Gene Expression Dynamics) model
    Implements a graph-based model for predicting gene expression trajectories
    with spatial and temporal context.
    """
    def __init__(
        self,
        num_genes,
        hidden_dim=64,
        num_gat_layers=1,
        num_mlp_layers=2,
        dropout=0.1,
        delta_gl=1,  # Time lag for gene -> ligand
        delta_lr=5,  # Time lag for ligand -> receptor 
        delta_rg=3,  # Time lag for receptor -> gene
        delta_gg=7,  # Time lag for gene -> gene
        add_self_loops=False,
        device=None
    ):
        super(STAGED, self).__init__()
        
        self.num_gat_layers = num_gat_layers
        self.num_mlp_layers = num_mlp_layers
        self.dropout = dropout
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.delta_gl = delta_gl
        self.delta_lr = delta_lr
        self.delta_rg = delta_rg
        self.delta_gg = delta_gg
        self.add_self_loops = add_self_loops
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initial feature dimensions
        self.input_dim = 1  # Single gene expression value
        
        # GAT layers
        assert num_gat_layers == 1, "Must have exactly one GAT layer"
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(self.input_dim, hidden_dim, heads=1, dropout=dropout, add_self_loops=add_self_loops)) # optionally use GATv2Conv? TODO investigate difference.
        
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, add_self_loops=add_self_loops))
        
        # MLP for prediction using Sequential
        mlp_layers = []
        mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_mlp_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
        
        mlp_layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, batch_data):
        """
        Forward pass of the STAGED model
        
        Args:
            batch_data: PyTorch Geometric Data or Batch object
                Can be a single graph or a batch of graphs
            
        Returns:
            node_embeddings: Node embeddings after GAT layers
            attention_weights: Attention weights from the last GAT layer
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        
        # Track attention weights from the last layer
        attention = None
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            # The GATConv automatically respects graph boundaries in batched data
            x, attention = gat_layer(x, edge_index, return_attention_weights=True) # only returns the last layer's attention weights
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        return x, attention
    
    def predict_genes(self, node_embeddings, gene_indices):
        """
        Generate predictions for gene nodes
        
        Args:
            node_embeddings: Embeddings for all nodes
            gene_indices: Indices of the gene nodes to predict
            
        Returns:
            predictions: Gene expression predictions [num_genes, 1]
        """
        # Get embeddings for gene nodes only
        gene_embeddings = node_embeddings[gene_indices]
        
        # Apply MLP to get predictions
        predictions = self.mlp(gene_embeddings)
        
        return predictions
        
    def get_t_init(self):
        """Return the initial time steps needed before prediction can start"""
        return max(self.delta_gl, self.delta_lr, self.delta_rg, self.delta_gg)

    def predict(
        self,
        data_processor,
        data: Dict[str, Any],
        time_point: int,
        cell_ids: Optional[List[int]] = None,
        store_attention: bool = False,
        batch_size: int = 32
    ) -> PredictionOutput:
        """
        Process raw data and generate predictions.
        
        Args:
            data: ProcessedData object containing input data
            time_point: Time point to predict for
            cell_ids: Optional list of cell IDs to predict for
            store_attention: Whether to store attention weights
            
        Returns:
            PredictionOutput containing predictions and optional attention weights
        """
        self.eval()
        with torch.no_grad():
            # Get input data
            gene_expression = data.gene_expression
            
            # If no cell_ids provided, predict for all cells
            if cell_ids is None:
                cell_ids = list(range(data.n_cells))
            
            # Initialize storage for predictions
            predictions = []
            attention_weights = [] if store_attention else None
            
            # Process each cell
            for cell_idx in cell_ids:
                # Process cell using data processor
                cell_predictions, cell_attention = data.process_cell(
                    gene_expression=gene_expression,
                    cell_positions=data.cell_positions,
                    time_point=time_point,
                    cell_idx=cell_idx,
                    delta_gl=self.delta_gl,
                    delta_lr=self.delta_lr,
                    delta_rg=self.delta_rg,
                    delta_gg=self.delta_gg,
                    store_attention=store_attention
                )
                
                predictions.append(cell_predictions)
                if store_attention and cell_attention is not None:
                    attention_weights.append(cell_attention)
            
            # Stack predictions
            predictions = torch.stack(predictions, dim=0)  # Shape: [n_cells, n_genes]
            predictions = predictions.unsqueeze(0)  # Shape: [1, n_cells, n_genes]
            
            if store_attention and attention_weights:
                attention_weights = torch.stack(attention_weights, dim=0)
            
            return PredictionOutput(
                predictions=predictions,
                attention_weights=attention_weights if store_attention else None
            )

    def predict_ode(
        self,
        data: ProcessedData,
        time_point: int,
        initial_state: Optional[torch.Tensor] = None,
        method: str = 'rk4',
        cell_ids: Optional[List[int]] = None,
        store_attention: bool = True,
        ode_func: Callable = None
    ) -> PredictionOutput:
        """
        Predict using Neural ODE integration.
        
        Args:
            data: ProcessedData object containing:
                - gene_expression: Historical gene expression for interpolation
                - cell_positions: Cell positions
                - n_cells: Number of cells
            time_point: Current time point
            initial_state: Initial condition (if None, uses gene_expression at time_point)
            method: ODE integration method
            cell_ids: Optional list of cell IDs to predict for
            store_attention: Whether to store attention weights during integration
            
        Returns:
            PredictionOutput with predictions and optionally attention weights
        """
        # Setup if not already done
        self.ode_func = ode_func

        # Get input data (already on device)
        gene_expression = data.gene_expression
        cell_positions = data.cell_positions
        
        # Store cell positions for ODE function access
        self._cell_positions = cell_positions
        
        # Set up temporary storage for attention data
        if store_attention:
            self._temp_attention_storage = []
            self._temp_pointer_storage = []
        else:
            self._temp_attention_storage = None
            self._temp_pointer_storage = None
        
        try:
            # Determine cells to process
            if cell_ids is None:
                cell_ids = list(range(data.n_cells))
            
            # Handle empty case
            if not cell_ids:
                raise ValueError("No cells to process")
            
            n_cells = len(cell_ids)
            
            # Set up initial condition
            if initial_state is None:
                # Extract initial state from gene expression at time_point
                initial_state = gene_expression[time_point, cell_ids, :].view(-1)
            
            initial_state = initial_state.detach().requires_grad_(True)
            
            # Create time span from current time_point to next time step
            current_time = float(time_point)
            next_time = current_time + 1.0
            time_span = torch.tensor([current_time, next_time], device=self.device)
            
            # Integrate ODE from current time to next time step
            solution = odeint_fixed(
                func=self.ode_func,
                y0=initial_state,
                t=time_span,
                method=method
            )
            
            # Take the prediction at the next time step (index 1)
            final_state = solution[-1]  # Shape: [n_cells * n_genes]

            predictions = final_state.view(1, n_cells, self.num_genes) # Shape: [1, n_cells, n_genes]

            # Collect attention data
            attention_weights = self._temp_attention_storage if store_attention else None
            node_pointers = self._temp_pointer_storage if store_attention else None
            
            return PredictionOutput(
                predictions=predictions,
                attention_weights=attention_weights,
                node_pointers=node_pointers
            )
            
        finally:
            # Clean up temporary storage
            if hasattr(self, '_temp_attention_storage'):
                delattr(self, '_temp_attention_storage')
            if hasattr(self, '_temp_pointer_storage'):
                delattr(self, '_temp_pointer_storage')

