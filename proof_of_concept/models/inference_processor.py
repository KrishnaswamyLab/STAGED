import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import networkx as nx

from models.staged import STAGED
from utils.graph_constructor import GraphConstructor
from utils.graph_data_handler import GraphDataHandler

@dataclass
class PredictionOutput:
    """Structured output from prediction"""
    predictions: torch.Tensor
    attention_weights: Tuple[torch.Tensor, torch.Tensor]  # (edges, values)
    node_pointers: torch.Tensor

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
        # print(cell_idx, time_point,gene_expression.shape, cell_positions.shape)
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

    # @torch.no_grad()
    def predict(
        self,
        data: Dict[str, Any],
        time_point: int,
        cell_ids: Optional[List[int]] = None
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
            # return PredictionOutput(
            #     predictions=torch.empty((0, self.num_genes), device=self.device),
            #     attention_weights=(torch.empty(0), torch.empty(0)),
            #     node_pointers=torch.empty(0)
            # )

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