import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np


class GraphConstructor:
    """
    Utility class for constructing cell-specific graphs for the STAGED model.
    It handles the creation of gene, ligand, and receptor nodes, as well as
    the connections between them.
    """
    def __init__(self, genes, ligand_receptor_pairs, cell_type_assignments, prior_grns):
        """
        Initialize the graph constructor
        
        Args:
            genes: List of gene identifiers
            ligand_receptor_pairs: List of (ligand, receptor) gene pairs
            cell_type_assignments: Dictionary mapping cell IDs to cell types
            prior_grns: Dictionary mapping cell types to prior GRNs (as networkx graphs)
        """
        self.genes = genes
        self.gene_indices = {gene: idx for idx, gene in enumerate(genes)}
        self.ligand_receptor_pairs = ligand_receptor_pairs
        self.cell_type_assignments = cell_type_assignments
        self.prior_grns = prior_grns
        
        # Identify receptor and ligand genes
        self.receptor_genes = set(receptor for _, receptor in ligand_receptor_pairs)
        self.ligand_genes = set(ligand for ligand, _ in ligand_receptor_pairs)
        
    def construct_base_graph(self, cell_id):
        """
        Construct a base graph for a cell based on its cell type
        
        Args:
            cell_id: Cell identifier
            
        Returns:
            base_graph: NetworkX graph with gene, receptor, and ligand nodes
        """
        cell_type = self.cell_type_assignments[cell_id]
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
            
            # Connect receptor node to all genes
            for target_gene in self.genes:
                base_grn.add_edge(receptor_node, target_gene)
                
        # Add ligand output nodes for ligand genes
        for gene in self.ligand_genes:
            ligand_node = f"l_{gene}"
            base_grn.add_node(ligand_node, type="ligand", gene=gene)
            
            # Connect ligand gene to its protein node
            base_grn.add_edge(gene, ligand_node)
            
        return base_grn
    
    def update_graph_with_neighbors(self, graph, cell_id, cell_positions, time_point,
                                   gene_expression_history, distance_threshold=10.0):
        """
        Update a cell's graph by adding connections to neighboring cells' ligands
        
        Args:
            graph: Base graph for the cell
            cell_id: Cell identifier
            cell_positions: Dictionary mapping cell IDs to spatial positions at each time point
            time_point: Current time point
            gene_expression_history: Expression history up to the current time point
            distance_threshold: Maximum distance to consider cells as neighbors
            
        Returns:
            updated_graph: Updated graph with neighbor connections
        """
        updated_graph = graph.copy()
        current_position = cell_positions[cell_id][time_point]
        
        # Find neighboring cells
        neighbor_cells = []
        for other_cell_id, positions in cell_positions.items():
            if other_cell_id == cell_id:
                continue
                
            if time_point in positions:
                other_position = positions[time_point]
                distance = np.linalg.norm(np.array(current_position) - np.array(other_position))
                
                if distance <= distance_threshold:
                    neighbor_cells.append(other_cell_id)
        
        # Add input ligand nodes from neighboring cells
        for neighbor_cell_id in neighbor_cells:
            for ligand_gene, receptor_gene in self.ligand_receptor_pairs:
                # Create a node for the ligand from the neighbor cell
                input_ligand_node = f"l_{neighbor_cell_id}_{ligand_gene}"
                updated_graph.add_node(input_ligand_node, type="input_ligand", 
                                      cell=neighbor_cell_id, gene=ligand_gene)
                
                # Connect this input ligand to the appropriate receptor
                receptor_node = f"r_{receptor_gene}"
                updated_graph.add_edge(input_ligand_node, receptor_node)
        
        return updated_graph
    
    def assign_node_features(self, graph, cell_id, time_point, gene_expression_history,
                            delta_gl, delta_lr, delta_rg, delta_gg):
        """
        Assign features to nodes with appropriate time lags
        
        Args:
            graph: Cell graph
            cell_id: Cell identifier
            time_point: Current time point
            gene_expression_history: Expression history up to the current time point
            delta_gl, delta_lr, delta_rg, delta_gg: Time lags for different connection types
            
        Returns:
            graph_data: PyTorch Geometric Data object with node features
        """
        # Create feature vector for each node
        node_features = {}
        gene_node_indices = []
        
        # Get a list of nodes once to avoid calling graph.nodes() multiple times
        # and to be able to index into it
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
                # Add robust checking
                if (expr_time >= 0 and 
                    cell_id in gene_expression_history and
                    gene_idx in gene_expression_history[cell_id] and
                    expr_time in gene_expression_history[cell_id][gene_idx]):
                    feature = gene_expression_history[cell_id][gene_idx][expr_time]
                else:
                    feature = 0.0
                    
                node_features[node] = [feature]
                
            elif node_type == 'ligand':
                # This is an output ligand node
                gene = graph.nodes[node]['gene']
                gene_idx = self.gene_indices[gene]
                
                # Use gene expression with gene-ligand time lag
                expr_time = time_point - delta_gl
                # Add robust checking
                if (expr_time >= 0 and
                    cell_id in gene_expression_history and
                    gene_idx in gene_expression_history[cell_id] and
                    expr_time in gene_expression_history[cell_id][gene_idx]):
                    feature = gene_expression_history[cell_id][gene_idx][expr_time]
                else:
                    feature = 0.0
                    
                node_features[node] = [feature]
                
            elif node_type == 'input_ligand':
                # This is an input ligand node from another cell
                neighbor_cell_id = graph.nodes[node]['cell']
                gene = graph.nodes[node]['gene']
                gene_idx = self.gene_indices[gene]
                
                # Use neighbor's gene expression with ligand-receptor time lag
                expr_time = time_point - delta_lr
                
                # More robust checking for neighbor's gene expression
                if (expr_time >= 0 and 
                    neighbor_cell_id in gene_expression_history and 
                    gene_idx in gene_expression_history[neighbor_cell_id] and
                    expr_time in gene_expression_history[neighbor_cell_id][gene_idx]):
                    feature = gene_expression_history[neighbor_cell_id][gene_idx][expr_time]
                else:
                    feature = 0.0
                    
                node_features[node] = [feature]
                
            elif node_type == 'receptor':
                # This is a receptor node
                gene = graph.nodes[node]['gene']
                gene_idx = self.gene_indices[gene]
                
                # Use gene expression with receptor-gene time lag
                expr_time = time_point - delta_rg
                # Add robust checking
                if (expr_time >= 0 and
                    cell_id in gene_expression_history and
                    gene_idx in gene_expression_history[cell_id] and
                    expr_time in gene_expression_history[cell_id][gene_idx]):
                    feature = gene_expression_history[cell_id][gene_idx][expr_time]
                else:
                    feature = 0.0
                    
                node_features[node] = [feature]
        
        # Convert to tensor features in the same order as node_list
        features = torch.tensor([node_features[node] for node in node_list], dtype=torch.float)
        
        # Create a PyTorch Geometric Data object manually instead of using from_networkx
        edge_index = []
        for src, dst in graph.edges():
            src_idx = node_list.index(src)
            dst_idx = node_list.index(dst)
            edge_index.append([src_idx, dst_idx])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Create the PyG Data object
        pyg_graph = Data(x=features, edge_index=edge_index)
        pyg_graph.gene_node_indices = gene_node_indices
        
        return pyg_graph 