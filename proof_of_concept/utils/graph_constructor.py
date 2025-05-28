import torch
import networkx as nx
from torch_geometric.data import Data
import numpy as np


class GraphConstructor:
    """
    Utility class for constructing cell-specific graphs for the STAGED model.
    It handles the creation of gene, ligand, and receptor nodes, as well as
    the connections between them.
    """
    def __init__(self, genes, ligand_receptor_pairs, receptor_gene_pairs, cell_type_assignments, prior_grns):
        """
        Initialize the graph constructor
        
        Args:
            genes: List of gene identifiers
            ligand_receptor_pairs: List of (ligand, receptor) gene pairs
            receptor_gene_pairs: List of (receptor, gene) pairs for selective connections
            cell_type_assignments: Tensor or list of shape (n_cells,) assigning cell types
            prior_grns: Dictionary mapping cell types to prior GRNs (as networkx graphs)
        """
        self.genes = genes
        self.gene_indices = {gene: idx for idx, gene in enumerate(genes)}
        self.ligand_receptor_pairs = ligand_receptor_pairs
        self.receptor_gene_pairs = receptor_gene_pairs
        self.cell_type_assignments = cell_type_assignments
        self.prior_grns = prior_grns
        
        # Identify receptor and ligand genes
        self.receptor_genes = set(receptor for _, receptor in ligand_receptor_pairs)
        self.ligand_genes = set(ligand for ligand, _ in ligand_receptor_pairs)
        
        # Create a mapping of which genes each receptor connects to
        self.receptor_targets = {}
        for receptor, target_gene in receptor_gene_pairs:
            if receptor not in self.receptor_targets:
                self.receptor_targets[receptor] = set()
            self.receptor_targets[receptor].add(target_gene)
        
    def construct_base_graph(self, cell_idx):
        """
        Construct a base graph for a cell based on its cell type
        
        Args:
            cell_idx: Cell index in the tensor
            
        Returns:
            base_graph: NetworkX graph with gene, receptor, and ligand nodes
        """
        # Get cell type from the 1D cell_type_assignments list or tensor
        cell_type = self.cell_type_assignments[cell_idx]
        # If cell_type_assignments is a tensor, convert to int or string as needed
        if isinstance(cell_type, torch.Tensor):
            cell_type = cell_type.item()
            
        base_grn = self.prior_grns[cell_type].copy()
        
        # Create a mapping from gene IDs to node indices in the graph
        node_mapping = {}
        
        # Add gene nodes first
        for gene_idx, gene in enumerate(self.genes):
            node_mapping[gene] = gene_idx
            if gene not in base_grn.nodes():
                base_grn.add_node(gene)
                
        # Add receptor nodes for receptor genes
        for gene in self.receptor_genes:
            receptor_node = f"r_{gene}"
            base_grn.add_node(receptor_node, type="receptor", gene=gene)
            
            # Connect receptor node only to specified target genes
            if gene in self.receptor_targets:
                for target_gene in self.receptor_targets[gene]:
                    base_grn.add_edge(receptor_node, target_gene)
            
            # connect receptor gene to its protein node
            base_grn.add_edge(gene, receptor_node)
                
        # Add ligand output nodes for ligand genes
        for gene in self.ligand_genes:
            ligand_node = f"l_{gene}"
            base_grn.add_node(ligand_node, type="ligand", gene=gene)
            
            # Connect ligand gene to its protein node
            base_grn.add_edge(gene, ligand_node)
            
        return base_grn
    
    def update_graph_with_neighbors(self, graph, cell_idx, cell_positions, time_point,
                                   distance_threshold=10.0):
        """
        Update a cell's graph by adding connections to neighboring cells' ligands
        
        Args:
            graph: networkx graph
                Base graph for the cell
            cell_idx: int
                Cell index in the tensor
            cell_positions: tensor of shape (n_time_points, n_cells, 2)
                Spatial positions of cells across time
            time_point: int
                Current time point index
            distance_threshold: float
                Maximum distance to consider cells as neighbors
            
        Returns:
            updated_graph: Updated graph with neighbor connections
        """
        updated_graph = graph.copy()
        current_position = cell_positions[time_point, cell_idx]
        
        # Extract all cell positions at this time point
        all_positions = cell_positions[time_point]
        
        # Calculate distances between current cell and all other cells
        distances = torch.norm(all_positions - current_position.unsqueeze(0), dim=1)
        
        # Find cells within the distance threshold
        neighbor_indices = torch.where(distances <= distance_threshold)[0]
        
        # Add input ligand nodes from neighboring cells
        for neighbor_idx in neighbor_indices:
            # Convert to Python scalar for use in string formatting
            neighbor_idx_val = neighbor_idx.item()
            
            for ligand_gene, receptor_gene in self.ligand_receptor_pairs:
                # Create a node for the ligand from the neighbor cell
                input_ligand_node = f"l_{neighbor_idx_val}_{ligand_gene}"
                updated_graph.add_node(input_ligand_node, type="input_ligand", 
                                      cell=neighbor_idx_val, gene=ligand_gene)
                
                # Connect this input ligand to the appropriate receptor
                receptor_node = f"r_{receptor_gene}"
                updated_graph.add_edge(input_ligand_node, receptor_node)
        
        return updated_graph
    
    def assign_node_features(self, graph, cell_idx, time_point, gene_expression_history,
                            delta_gl, delta_lr, delta_rg, delta_gg):
        """
        Assign features to nodes with appropriate time lags
        
        Args:
            graph: networkx graph
                Cell graph
            cell_idx: int
                Cell index in the tensor
            time_point: int
                Current time point index
            gene_expression_history: tensor of shape (n_time_points, n_cells, n_genes)
                Expression history of all cells up to the current time point
            delta_gl, delta_lr, delta_rg, delta_gg: int
                Time lags for different connection types
            
        Returns:
            graph_data: PyTorch Geometric Data object with node features
        """
        # Create feature vector for each node
        node_features = {}
        gene_node_indices = []
        
        # Get a list of nodes once to avoid calling graph.nodes() multiple times
        node_list = list(graph.nodes())
        
        for i, node in enumerate(node_list):
            node_type = graph.nodes[node].get('type', 'gene')
            
            if node_type == 'gene':
                # This is a gene node
                gene = node
                gene_idx = self.gene_indices[gene]
                # Store the index of this node in the node list
                gene_node_indices.append(i)
                
                # Use gene expression with gene-gene time lag
                expr_time = time_point - delta_gg
                
                # Check if the time point is valid (≥ 0)
                if expr_time >= 0:
                    # Access the tensor directly with [time, cell, gene] indexing
                    feature = gene_expression_history[expr_time, cell_idx, gene_idx].item()
                else:
                    # Fix the exception raising syntax
                    raise ValueError(f'Expression history is not long enough given time lag {delta_gg}')
                    
                node_features[node] = [feature]
                
            elif node_type == 'ligand':
                # This is an output ligand node
                gene = graph.nodes[node]['gene']
                gene_idx = self.gene_indices[gene]
                
                # Use gene expression with gene-ligand time lag
                expr_time = time_point - delta_gl
                
                # Check if the time point is valid
                if expr_time >= 0:
                    feature = gene_expression_history[expr_time, cell_idx, gene_idx].item()
                else:
                    raise ValueError(f'Expression history is not long enough given time lag {delta_gl}')
                    
                node_features[node] = [feature]
                
            elif node_type == 'input_ligand':
                # This is an input ligand node from another cell
                neighbor_cell_idx = graph.nodes[node]['cell']
                gene = graph.nodes[node]['gene']
                gene_idx = self.gene_indices[gene]
                
                # Use neighbor's gene expression with ligand-receptor time lag
                expr_time = time_point - delta_lr
                
                # Check if the time point is valid
                if expr_time >= 0:
                    feature = gene_expression_history[expr_time, neighbor_cell_idx, gene_idx].item()
                else:
                    raise ValueError(f'Expression history is not long enough given time lag {delta_lr}')
                    
                node_features[node] = [feature]
                
            elif node_type == 'receptor':
                # This is a receptor node
                gene = graph.nodes[node]['gene']
                gene_idx = self.gene_indices[gene]
                
                # Use gene expression with receptor-gene time lag
                expr_time = time_point - delta_rg
                
                # Check if the time point is valid
                if expr_time >= 0:
                    feature = gene_expression_history[expr_time, cell_idx, gene_idx].item()
                else:
                    raise ValueError(f'Expression history is not long enough given time lag {delta_rg}')
                    
                node_features[node] = [feature]
        
        # Convert to tensor features in the same order as node_list
        features = torch.tensor([node_features[node] for node in node_list], dtype=torch.float)
        
        # Create edge indices for the PyG Data object
        edge_index = []
        for src, dst in graph.edges():
            src_idx = node_list.index(src)
            dst_idx = node_list.index(dst)
            edge_index.append([src_idx, dst_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create the PyG Data object
        pyg_graph = Data(x=features, edge_index=edge_index)
        pyg_graph.gene_node_indices = gene_node_indices
        
        # Store node names and types
        pyg_graph.node_names = node_list  # Store the ordered list of node names
        pyg_graph.node_types = [graph.nodes[node].get('type', 'gene') for node in node_list]  # Store node types
        
        return pyg_graph

    def assign_node_features_ode(self, graph, cell_idx_in_dataset: int, current_ode_time_t: float,
                                 current_y_for_cell: torch.Tensor, 
                                 history_interpolator, 
                                 delta_gl: int, delta_lr: int, delta_rg: int, delta_gg: int,
                                 device: torch.device = torch.device('cpu')):
        """
        Assign features to nodes for Neural ODE mode.
        - 'gene' nodes use current_y_for_cell (delta_gg is ignored)
        - Other nodes use history_interpolator with their respective deltas
        
        Args:
            graph: networkx graph for the cell
            cell_idx_in_dataset: Original index of the cell in the dataset
            current_ode_time_t: Current time t from ODE solver
            current_y_for_cell: Current ODE state for this cell's genes (n_genes,)
            history_interpolator: HistoryInterpolator instance
            delta_gl, delta_lr, delta_rg, delta_gg: Time lags (delta_gg ignored for gene nodes)
            device: torch device for output tensors
            
        Returns:
            graph_data: PyTorch Geometric Data object with node features
        """
        if delta_gg != 0:
            print(f"Warning: delta_gg={delta_gg} is specified but will be ignored for 'gene' nodes in ODE mode.")

        node_features = {}
        gene_node_indices = []
        node_list = list(graph.nodes())

        for i, node in enumerate(node_list):
            node_type = graph.nodes[node].get('type', 'gene')
            feature_val = 0.0

            if node_type == 'gene':
                gene_id = node
                gene_idx = self.gene_indices[gene_id]
                gene_node_indices.append(i)
                # Use current ODE state for gene nodes
                feature_val = current_y_for_cell[gene_idx].item()
                
            elif node_type == 'ligand':
                gene = graph.nodes[node]['gene']
                gene_idx = self.gene_indices[gene]
                # Use interpolated history with gene-ligand lag
                expr_time = current_ode_time_t - delta_gl
                feature_val = history_interpolator.interpolate(expr_time, cell_idx_in_dataset, gene_idx)

            elif node_type == 'input_ligand':
                # Get neighbor cell index from the 'cell' attribute (set by update_graph_with_neighbors)
                neighbor_cell_idx = graph.nodes[node]['cell']
                gene = graph.nodes[node]['gene']
                gene_idx = self.gene_indices[gene]
                # Use interpolated history with ligand-receptor lag
                expr_time = current_ode_time_t - delta_lr
                feature_val = history_interpolator.interpolate(expr_time, neighbor_cell_idx, gene_idx)

            elif node_type == 'receptor':
                gene = graph.nodes[node]['gene']
                gene_idx = self.gene_indices[gene]
                # Use interpolated history with receptor-gene lag
                expr_time = current_ode_time_t - delta_rg
                feature_val = history_interpolator.interpolate(expr_time, cell_idx_in_dataset, gene_idx)

            node_features[node] = [feature_val]

        # Convert to tensors
        features = torch.tensor([node_features[node] for node in node_list], dtype=torch.float, device=device)
        
        # Create edge indices
        edge_list = []
        for src, dst in graph.edges():
            src_idx = node_list.index(src)
            dst_idx = node_list.index(dst)
            edge_list.append([src_idx, dst_idx])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        # Create PyG Data object
        pyg_graph = Data(x=features, edge_index=edge_index)
        pyg_graph.gene_node_indices = torch.tensor(gene_node_indices, dtype=torch.long, device=device)
        
        return pyg_graph 