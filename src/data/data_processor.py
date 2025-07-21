import torch
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from torch_geometric.data import Data, Batch
from src.utils.graph_constructor import GraphConstructor
from src.utils.graph_data_handler import GraphDataHandler

@dataclass
class ProcessedData:
    """Container for processed data"""
    gene_expression: torch.Tensor
    cell_positions: torch.Tensor
    cell_type_assignments: torch.Tensor
    ligand_receptor_pairs: List[tuple]
    receptor_gene_pairs: List[tuple]
    prior_grns: Dict[Any, Any]
    n_cells: int
    n_time_points: int
    genes: List[str]
    num_genes: int

class DataProcessor:
    ''''
    Processes single-cell gene expression data into graph representations for gene regulatory network analysis.
    
    Converts gene expression, cell positions, and regulatory relationships into graph structures
    that capture both intra-cellular gene regulation and inter-cellular communication through
    ligand-receptor interactions.
    '''

    def __init__(
        self,
        genes: List[str],
        ligand_receptor_pairs: List[tuple],
        receptor_gene_pairs: List[tuple],
        cell_type_assignments: Any,
        prior_grns: Dict[Any, Any],
        device: torch.device,
        distance_threshold: float = 1.0,
        batch_size: int = 32,
        model=None
    ):
        self.genes = genes
        self.ligand_receptor_pairs = ligand_receptor_pairs
        self.receptor_gene_pairs = receptor_gene_pairs
        self.cell_type_assignments = cell_type_assignments
        self.prior_grns = prior_grns
        self.device = device
        self.distance_threshold = distance_threshold
        self.num_genes = len(genes)
        self.batch_size = batch_size
        self.model = model
        
        # Initialize graph constructor and handler
        self.graph_constructor = GraphConstructor(
            genes=genes,
            ligand_receptor_pairs=ligand_receptor_pairs,
            receptor_gene_pairs=receptor_gene_pairs,
            cell_type_assignments=cell_type_assignments,
            prior_grns=prior_grns
        )
        self.graph_handler = GraphDataHandler(model, device=device)
        
        # ODE-specific attributes
        self._ode_func = None
        self.ode_integration_const = None

        if not genes:
            raise ValueError("Empty genes list provided to DataProcessor")
        if not isinstance(genes, list):
            raise ValueError(f"genes must be a list, got {type(genes)}")
    
        print(f"DataProcessor initialized with {len(genes)} genes")
        self.genes = genes
        self.num_genes = len(genes)
    
        if self.num_genes == 0:
            raise ValueError("num_genes is 0 after initialization")


    def preprocess_data(self, data: Dict[str, torch.Tensor]) -> ProcessedData:
        """Preprocess raw data into model-ready format."""
        processed_data = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }
        self._cell_positions = processed_data['cell_positions']

        return ProcessedData(
            gene_expression=processed_data['gene_expression'],
            cell_positions=processed_data['cell_positions'],
            cell_type_assignments=processed_data['cell_type_assignments'],
            ligand_receptor_pairs=self.ligand_receptor_pairs,
            receptor_gene_pairs=self.receptor_gene_pairs,
            prior_grns=self.prior_grns,
            n_cells=processed_data['n_cells'],
            n_time_points=processed_data['n_time_points'],
            genes=self.genes,
            num_genes=self.num_genes
        )

    def get_cell_neighbors(self, cell_positions: torch.Tensor) -> torch.Tensor:
        """
        Compute cell neighborhood relationships based on spatial positions.
        
        Args:
            cell_positions: Tensor of shape (n_cells, 2) containing cell positions
            
        Returns:
            Tensor of shape (n_cells, n_cells) containing neighborhood relationships
        """
        # Compute pairwise distances
        dist = torch.cdist(cell_positions, cell_positions)
        
        # Create adjacency matrix based on distance threshold
        adj = (dist <= self.distance_threshold).float()
        
        # Remove self-loops
        adj.fill_diagonal_(0)
        
        return adj

    def construct_graph_ode(
        self,
        cell_idx: int,
        current_ode_time_t: float,
        current_y_for_cell: torch.Tensor,
        delta_gl: int,
        delta_lr: int,
        delta_rg: int,
        delta_gg: int
    ) -> Data:
        """
        Construct a graph for ODE mode using the GraphConstructor.
        
        Args:
            cell_idx: Index of the cell to construct graph for
            current_ode_time_t: Current time from ODE solver
            current_y_for_cell: Current ODE state for this cell's genes
            delta_gl: Time lag for gene-ligand connections
            delta_lr: Time lag for ligand-receptor connections
            delta_rg: Time lag for receptor-gene connections
            delta_gg: Time lag for gene-gene connections
            
        Returns:
            PyTorch Geometric Data object containing the graph
        """
        # Construct base graph
        base_graph = self.graph_constructor.construct_base_graph(cell_idx)
        
        # Assign node features for ODE mode
        graph_data = self.graph_constructor.assign_node_features_ode(
            graph=base_graph,
            cell_idx_in_dataset=cell_idx,
            current_ode_time_t=current_ode_time_t,
            current_y_for_cell=current_y_for_cell,
            delta_gl=delta_gl,
            delta_lr=delta_lr,
            delta_rg=delta_rg,
            delta_gg=delta_gg,
            device=self.device
        )
        
        return graph_data

    def construct_batch(
        self,
        gene_expression: torch.Tensor,
        cell_positions: torch.Tensor,
        cell_ids: Optional[List[int]] = None,
        batch_size: int = 32
    ) -> Batch:
        """
        Construct a batch of graphs for training.
        
        Args:
            gene_expression: Tensor of shape (n_cells, n_genes) containing gene expression
            cell_positions: Tensor of shape (n_cells, 2) containing cell positions
            cell_ids: Optional list of cell IDs to include in the graphs
            batch_size: Number of cells per batch
            
        Returns:
            PyTorch Geometric Batch object containing the batched graphs
        """
        # Filter data if cell_ids provided
        if cell_ids is not None:
            gene_expression = gene_expression[cell_ids]
            cell_positions = cell_positions[cell_ids]
        
        # Create list to store individual graphs
        graphs = []
        
        # Create graphs for each batch
        for i in range(0, len(cell_ids) if cell_ids is not None else gene_expression.shape[0], batch_size):
            batch_cell_ids = list(range(i, min(i + batch_size, gene_expression.shape[0])))
            graph = self.construct_graph(gene_expression, cell_positions, 0, batch_cell_ids[0], 0, 0, 0, 0)
            graphs.append(graph)
        
        # Create batch
        return Batch.from_data_list(graphs)

    
    def process_cell_data_ode(
        self,
        cell_idx: int,
        t: float,
        current_y_for_cell: torch.Tensor,
        cell_positions: torch.Tensor,
  
    ) -> torch.Tensor:
        """
        Process a single cell's data into graph format for ODE mode.
        
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
            base_graph, 
            cell_idx,
            cell_positions, 
            latest_time_idx,
            distance_threshold=self.distance_threshold
        )
    
        return self.graph_constructor.assign_node_features_ode(
            updated_graph, cell_idx, t, current_y_for_cell,
            self.model.delta_gl, self.model.delta_lr,
            self.model.delta_rg, self.model.delta_gg
        )
    
    def ode_func(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """
        ODE function: dy/dt = f(t, y)
        
        Args:
            t: Current time (float)
            y: Current state (n_cells * n_genes,)
            
        Returns:
            dy_dt: Derivatives (n_cells * n_genes,)
        """
        if self.num_genes == 0:
            raise ValueError(f"num_genes is 0! Check genes initialization. Current genes: {getattr(self, 'genes', 'NOT SET')}")
        # Determine number of cells from y shape
        concatenated_genes = y.shape[0] # all genes concatenated
        n_cells = concatenated_genes // self.num_genes
        if concatenated_genes % self.num_genes != 0:
            raise ValueError(f"State size {concatenated_genes} not divisible by num_genes {self.num_genes}")
        
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