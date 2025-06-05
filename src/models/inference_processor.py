import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import networkx as nx

from src.models.staged import STAGED
from src.utils.graph_constructor import GraphConstructor
from src.utils.graph_data_handler import GraphDataHandler
from src.utils.interpolation import HistoryInterpolator

from src.utils.ode import odeint_fixed

from torchdiffeq import odeint


@dataclass
class PredictionOutput:
    """Structured output from prediction"""
    predictions: torch.Tensor
    attention_weights: Tuple[torch.Tensor, torch.Tensor]  # (edges, values)
    node_pointers: torch.Tensor

@dataclass
class ODEPredictionOutput:
    """Structured output from ODE prediction"""
    predictions: torch.Tensor  # Shape: (n_eval_times, n_cells, n_genes)
    eval_times: torch.Tensor   # Shape: (n_eval_times,)
    attention_weights: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None  # List of attention weights for each time step
    node_pointers: Optional[List[torch.Tensor]] = None  # List of node pointers for each time step

class STAGEDProcessor:
    """
    A processor/coordinator class that handles the end-to-end inference pipeline:
    1. Data preprocessing
    2. Graph construction
    3. Feature assignment
    4. Model inference

    This class coordinates between different components (STAGED model, GraphConstructor,
    GraphDataHandler) to process raw cell data into predictions.
    """
    def __init__(
        self,
        model: STAGED,  # Pass in pre-trained model
        genes: List[str],
        ligand_receptor_pairs: List[tuple],
        receptor_gene_pairs: List[tuple],
        cell_type_assignments: Any,
        prior_grns: Dict[Any, nx.DiGraph],
        batch_size: int = 32,
        distance_threshold: float = 10.0,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the STAGED processor.

        Args:
            model: Pre-trained STAGED model
            genes: List of gene identifiers
            ligand_receptor_pairs: List of (ligand, receptor) gene pairs
            receptor_gene_pairs: List of (receptor, gene) pairs for selective connections
            cell_type_assignments: Cell type assignments (tensor or list)
            prior_grns: Dictionary mapping cell types to prior GRNs (networkx graphs)
            batch_size: Batch size for internal processing
            distance_threshold: Maximum distance for considering cell neighbors
            device: Device to run computations on (default: cuda if available, else cpu)
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()  # Ensure model is in eval mode
        
        self.batch_size = batch_size
        self.distance_threshold = distance_threshold
        self.num_genes = len(genes)

        # Initialize helpers
        self.graph_constructor = GraphConstructor(
            genes=genes,
            ligand_receptor_pairs=ligand_receptor_pairs,
            receptor_gene_pairs=receptor_gene_pairs,
            cell_type_assignments=cell_type_assignments,
            prior_grns=prior_grns
        )
        self.graph_handler = GraphDataHandler(self.model, device=self.device)
        
        # ODE-specific attributes (initialized when needed)
        self._ode_func = None        # Create ODE function
        self.ode_integration_const = None


    def process_cell_data(
        self,
        cell_idx: int,
        time_point: int,
        gene_expression: torch.Tensor,
        cell_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Process single cell data into graph format.
        
        Args:
            cell_idx: Index of the cell to process
            time_point: Current time point
            gene_expression: Gene expression tensor
            cell_positions: Cell positions tensor
            
        Returns:
            Processed graph data in PyG format
        """
        # Validate cell index
        if cell_idx >= gene_expression.shape[1] or cell_idx < 0:
            raise ValueError(f"Invalid cell_idx {cell_idx}. Must be between 0 and {gene_expression.shape[1]-1}")

        # Check time point validity
        min_required_time = max(
            self.model.delta_gl, self.model.delta_lr,
            self.model.delta_rg, self.model.delta_gg
        )
        if time_point < min_required_time:
            raise ValueError(
                f"time_point {time_point} is too early. Need at least "
                f"{min_required_time} time points of history for lags."
            )

        # Construct and process graph
        base_graph = self.graph_constructor.construct_base_graph(cell_idx)
        updated_graph = self.graph_constructor.update_graph_with_neighbors(
            base_graph, cell_idx, cell_positions, time_point,
            distance_threshold=self.distance_threshold
        )
        return self.graph_constructor.assign_node_features(
            updated_graph, cell_idx, time_point, gene_expression,
            self.model.delta_gl, self.model.delta_lr,
            self.model.delta_rg, self.model.delta_gg
        )

    def process_cell_data_ode(
        self,
        cell_idx: int,
        t: float,
        current_y_for_cell: torch.Tensor,
        cell_positions: torch.Tensor,
  
    ) -> torch.Tensor:
        """
        Process single cell data into graph format for ODE mode.
        
        Args:
            cell_idx: Index of the cell to process
            t: Current continuous time
            current_y_for_cell: Current ODE state for this cell's genes
            cell_positions: Cell positions tensor
            store_attention: Whether to store attention weights during integration
        Returns:
            Processed graph data in PyG format
        """

        # Construct and process graph
        base_graph = self.graph_constructor.construct_base_graph(cell_idx)
        
        # Use latest time point for cell positions
        latest_time_idx = cell_positions.shape[0] - 1
        updated_graph = self.graph_constructor.update_graph_with_neighbors(
            base_graph, cell_idx, cell_positions, latest_time_idx,
            distance_threshold=self.distance_threshold
        )
    
        return self.graph_constructor.assign_node_features_ode(
            updated_graph, cell_idx, t, current_y_for_cell,
            self.model.delta_gl, self.model.delta_lr,
            self.model.delta_rg, self.model.delta_gg
        )

    def predict(
        self,
        data: Dict[str, Any],
        time_point: int,
        cell_ids: Optional[List[int]] = None,
        store_attention: bool = False
    ) -> PredictionOutput:
        """
        Process raw data and generate predictions.
        
        Args:
            data: Dictionary containing:
                - gene_expression: Full historical gene expression tensor
                - cell_positions: Full historical cell position tensor
                - n_cells: Total number of cells (optional if cell_ids provided)
            time_point: Current time point
            cell_ids: Optional list of cell IDs to predict for (if None, predict for all cells)
            store_attention: Whether to store attention weights during integration
        Returns:
            PredictionOutput containing:
                - predictions: Predicted gene expression for next time step
                - attention_weights: GAT attention weights
                - node_pointers: Node batch assignments
        """
        # Move data to device
        gene_expression = data['gene_expression'].to(self.device)
        cell_positions = data['cell_positions'].to(self.device)

        # Determine cells to process
        if cell_ids is None:
            if 'n_cells' not in data:
                raise KeyError("'n_cells' missing from data when cell_ids not provided")
            cell_ids = range(data['n_cells'])

        # Process each cell
        cell_graphs = []
        for cell_idx in cell_ids:
            graph_data = self.process_cell_data(
                cell_idx, time_point, gene_expression, cell_positions
            )
            cell_graphs.append(graph_data.to(self.device))

        # Handle empty case
        if not cell_graphs:
            raise ValueError("No cells to process")

        # Get predictions using handler
        preds, attn, ptrs = self.graph_handler.process_cell_graphs(
            cell_graphs=cell_graphs,
            num_genes=self.num_genes,
            batch_size=self.batch_size
        )

        return PredictionOutput(preds, attn, ptrs)

    def predict_at_time(self, *args, **kwargs) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Backwards compatibility wrapper for predict().
        Returns tuple instead of PredictionOutput for compatibility.
        """
        output = self.predict(*args, **kwargs)
        return output.predictions, output.attention_weights, output.node_pointers

    def _create_ode_function(self):
        """Create the ODE function that will be used by torchdiffeq."""

        def ode_func(t: float, y: torch.Tensor) -> torch.Tensor:
            """
            ODE function: dy/dt = f(t, y)
            
            Args:
                t: Current time (float)
                y: Current state (n_cells * n_genes,)
                
            Returns:
                dy_dt: Derivatives (n_cells * n_genes,)
            """
            # Determine number of cells from y shape
            total_genes = y.shape[0]
            n_cells = total_genes // self.num_genes
            
            if total_genes % self.num_genes != 0:
                raise ValueError(f"State size {total_genes} not divisible by num_genes {self.num_genes}")
            
            # Reshape y to (n_cells, n_genes)
            y_reshaped = y.view(n_cells, self.num_genes)
            
            # Process each cell and collect derivatives
            cell_graphs = []
            for cell_idx in range(n_cells):
                # Get current state for this cell
                current_y_for_cell = y_reshaped[cell_idx]
                
                # Create graph with ODE features
                graph_data = self.process_cell_data_ode(
                    cell_idx=cell_idx,
                    t=t,
                    current_y_for_cell=current_y_for_cell,
                    cell_positions=self._cell_positions  # Stored during predict_ode call
                )
                cell_graphs.append(graph_data.to(self.device))
            
            # Get derivatives from model
            derivatives, attn, ptrs = self.graph_handler.process_cell_graphs(
                cell_graphs=cell_graphs,
                num_genes=self.num_genes,
                batch_size=self.batch_size
            )
            
            # Store attention weights and node pointers if storage exists
            if hasattr(self, '_temp_attention_storage') and self._temp_attention_storage is not None:
                self._temp_attention_storage.append(attn)
            if hasattr(self, '_temp_pointer_storage') and self._temp_pointer_storage is not None:
                self._temp_pointer_storage.append(ptrs)
            
            # Flatten back to original shape
            return derivatives.view(-1)
        
        return ode_func

    def predict_ode_new(
        self,
        data: Dict[str, Any],
        time_point: int,
        initial_state: Optional[torch.Tensor] = None,
        method: str = 'rk4',
        cell_ids: Optional[List[int]] = None,
        store_attention: bool = True
    ) -> PredictionOutput:
        """
        Predict using Neural ODE integration.
        
        Args:
            data: Dictionary containing:
                - gene_expression: Historical gene expression for interpolation
                - cell_positions: Cell positions
                - n_cells: Number of cells (optional if cell_ids provided)
            time_point: Current time point
            initial_state: Initial condition (if None, uses gene_expression at time_point)
            method: ODE integration method
            cell_ids: Optional list of cell IDs to predict for
            store_attention: Whether to store attention weights during integration
            
        Returns:
            PredictionOutput with predictions and optionally attention weights
        """
        # Setup if not already done
        self._ode_func = self._create_ode_function()

        # Move data to device
        gene_expression = data['gene_expression'].to(self.device)
        cell_positions = data['cell_positions'].to(self.device)
        
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
                if 'n_cells' not in data:
                    raise KeyError("'n_cells' missing from data when cell_ids not provided")
                cell_ids = list(range(data['n_cells']))
            
            # Handle empty case
            if not cell_ids:
                raise ValueError("No cells to process")
            
            n_cells = len(cell_ids)
            
            # Set up initial condition
            if initial_state is None:
                # Extract initial state from gene expression at time_point
                initial_state = gene_expression[time_point, cell_ids, :].view(-1)
            
            initial_state = initial_state.to(self.device).detach().requires_grad_(True)
            
            # Create time span from current time_point to next time step
            current_time = float(time_point)
            next_time = current_time + 1.0
            time_span = torch.tensor([current_time, next_time], device=self.device)
            
            # Integrate ODE from current time to next time step
            solution = odeint_fixed(
                func=self._ode_func,
                y0=initial_state,
                t=time_span,
                method=method
            )
            
            # Take the prediction at the next time step (index 1)
            final_state = solution[-1]  # Shape: [n_cells * n_genes]

            ##TODO: UNDERSTAND THE CONSTANT OVERSHOOT WE HAVE USING THE ODE
            predictions = final_state.view(1, n_cells, self.num_genes) - 1.5*torch.ones((1, n_cells, self.num_genes)) # Shape: [1, n_cells, n_genes]
            
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