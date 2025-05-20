import unittest
import os
import sys
import torch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.graph_constructor import GraphConstructor


class TestGraphConstructor(unittest.TestCase):

    def test_initialization(self):
        """Test that the GraphConstructor initializes correctly"""
        # Check receptor and ligand genes are identified
        self.assertEqual(self.graph_constructor.receptor_genes, {"gene_1", "gene_3"})
        self.assertEqual(self.graph_constructor.ligand_genes, {"gene_0", "gene_2"})
        
        # Check gene indices
        self.assertEqual(self.graph_constructor.gene_indices["gene_0"], 0)
        self.assertEqual(self.graph_constructor.gene_indices["gene_4"], 4)
    
    def test_construct_base_graph(self):
        """Test base graph construction"""
        # Construct a base graph for cell_0
        base_graph = self.graph_constructor.construct_base_graph("cell_0")
        
        # Check all genes are included
        for gene in self.genes:
            self.assertIn(gene, base_graph.nodes())
        
        # Check receptor nodes are created and connected to all genes
        for receptor_gene in self.graph_constructor.receptor_genes:
            receptor_node = f"r_{receptor_gene}"
            self.assertIn(receptor_node, base_graph.nodes())
            
            # Receptor should be connected to all genes
            for gene in self.genes:
                self.assertTrue(base_graph.has_edge(receptor_node, gene))
        
        # Check ligand nodes are created and connected to their genes
        for ligand_gene in self.graph_constructor.ligand_genes:
            ligand_node = f"l_{ligand_gene}"
            self.assertIn(ligand_node, base_graph.nodes())
            
            # Ligand should be connected to its gene
            self.assertTrue(base_graph.has_edge(ligand_gene, ligand_node))
    
    def test_update_graph_with_neighbors(self):
        """Test graph updating with neighbor information"""
        # First construct a base graph
        base_graph = self.graph_constructor.construct_base_graph("cell_0")
        
        # Update the graph with neighbor information
        time_point = 2
        updated_graph = self.graph_constructor.update_graph_with_neighbors(
            graph=base_graph,
            cell_id="cell_0",
            cell_positions=self.cell_positions,
            time_point=time_point,
            gene_expression_history=self.gene_expression_history,
            distance_threshold=50.0  # Large threshold to ensure neighbors
        )
        
        # Check that input ligand nodes from neighbors are added
        for neighbor_id in ["cell_1", "cell_2"]:
            for ligand_gene in self.graph_constructor.ligand_genes:
                input_ligand_node = f"l_{neighbor_id}_{ligand_gene}"
                self.assertIn(input_ligand_node, updated_graph.nodes())
                
                # Check that input ligand is connected to appropriate receptor
                for _, receptor_gene in self.lr_pairs:
                    if ligand_gene == self.lr_pairs[0][0] and receptor_gene == self.lr_pairs[0][1]:
                        receptor_node = f"r_{receptor_gene}"
                        self.assertTrue(updated_graph.has_edge(input_ligand_node, receptor_node))
                    elif ligand_gene == self.lr_pairs[1][0] and receptor_gene == self.lr_pairs[1][1]:
                        receptor_node = f"r_{receptor_gene}"
                        self.assertTrue(updated_graph.has_edge(input_ligand_node, receptor_node))
    
    def test_assign_node_features(self):
        """Test feature assignment to nodes"""
        # First construct a base graph and update with neighbors
        base_graph = self.graph_constructor.construct_base_graph("cell_0")
        time_point = 3
        updated_graph = self.graph_constructor.update_graph_with_neighbors(
            graph=base_graph,
            cell_id="cell_0",
            cell_positions=self.cell_positions,
            time_point=time_point,
            gene_expression_history=self.gene_expression_history,
            distance_threshold=50.0  # Large threshold to ensure neighbors
        )
        
        # Assign features with all time lags = 1
        delta_gl = delta_lr = delta_rg = delta_gg = 1
        pyg_graph = self.graph_constructor.assign_node_features(
            graph=updated_graph,
            cell_id="cell_0",
            time_point=time_point,
            gene_expression_history=self.gene_expression_history,
            delta_gl=delta_gl,
            delta_lr=delta_lr,
            delta_rg=delta_rg,
            delta_gg=delta_gg
        )
        
        # Check that PyTorch Geometric Data object is returned
        self.assertIsInstance(pyg_graph.x, torch.Tensor)
        
        # Check feature dimensions
        # Number of nodes should match number of features
        self.assertEqual(len(updated_graph.nodes()), pyg_graph.x.shape[0])
        
        # Each feature should be 1-dimensional (scalar)
        self.assertEqual(pyg_graph.x.shape[1], 1)
        
        # Check that gene_node_indices are stored
        self.assertTrue(hasattr(pyg_graph, 'gene_node_indices'))
        self.assertEqual(len(pyg_graph.gene_node_indices), len(self.genes))


def create_square_grid_data():
    """Create toy data for testing the GraphConstructor"""
    # Define dimensions - increase time points to handle larger lags
    n_time_points = 15  # Increased from 5 to accommodate larger time lags
    n_cells = 4
    n_genes = 6
    
    # Create gene expression tensor: (n_time_points, n_cells, n_genes)
    # For better testing, use values that clearly relate to time, cell, and gene
    gene_expression = torch.zeros((n_time_points, n_cells, n_genes))
    for t in range(n_time_points):
        for c in range(n_cells):
            for g in range(n_genes):
                # Create values that encode time, cell, and gene for easy verification
                # Format: t*100 + c*10 + g (e.g., time 2, cell 1, gene 3 = 213)
                gene_expression[t, c, g] = t*100 + c*10 + g
    
    # Create spatial positions tensor: (n_time_points, n_cells, 2)
    # Arrange cells in a 2x2 grid pattern
    cell_positions = torch.zeros((n_time_points, n_cells, 2))
    for t in range(n_time_points):
        cell_positions[t, 0] = torch.tensor([0.0, 0.0])    # Cell 0: top-left
        cell_positions[t, 1] = torch.tensor([10.0, 0.0])   # Cell 1: top-right
        cell_positions[t, 2] = torch.tensor([0.0, 10.0])   # Cell 2: bottom-left
        cell_positions[t, 3] = torch.tensor([10.0, 10.0])  # Cell 3: bottom-right
    
    # Define gene names
    genes = [f"gene_{i}" for i in range(n_genes)]
    
    # Define cell type assignments: cells 0,1 as type 0 and cells 2,3 as type 1
    cell_type_assignments = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    
    # Define ligand-receptor pairs
    ligand_receptor_pairs = [
        ("gene_0", "gene_1"),  # gene_0 is ligand, gene_1 is receptor
        ("gene_2", "gene_3"),  # gene_2 is ligand, gene_3 is receptor
    ]
    
    # Define receptor-gene pairs (selective connections)
    receptor_gene_pairs = [
        ("gene_1", "gene_4"),  # receptor gene_1 regulates gene_4
        ("gene_1", "gene_5"),  # receptor gene_1 also regulates gene_5
        ("gene_3", "gene_1"),  # receptor gene_3 regulates gene_1
        ("gene_3", "gene_4"),  # receptor gene_3 also regulates gene_4
        ("gene_4", "gene_2"),  # gene_4 regulates gene_2
        ("gene_4", "gene_3"),  # gene_4 regulates gene_3
        ("gene_5", "gene_3"),  # gene_5 regulates gene_3
        ("gene_5", "gene_0"),  # gene_5 regulates gene_0
    ]
    
    # Create prior gene regulatory networks (GRNs) for each cell type
    # GRN for cell type 0
    prior_grn_0 = nx.DiGraph()
    for i in range(n_genes):
        prior_grn_0.add_node(f"gene_{i}")
    # Add regulatory edges for cell type 0
    prior_grn_0.add_edge("gene_4", "gene_2")  # gene_4 regulates gene_2
    prior_grn_0.add_edge("gene_4", "gene_3")  # gene_4 regulates gene_3
    prior_grn_0.add_edge("gene_5", "gene_3")  # gene_5 regulates gene_3
    prior_grn_0.add_edge("gene_5", "gene_0")  # gene_5 regulates gene_0
    prior_grn_0.add_edge("gene_1", "gene_5")  # gene_1 regulates gene_5
    prior_grn_0.add_edge("gene_2", "gene_4")  # gene_2 regulates gene_4
    
    # GRN for cell type 1
    prior_grn_1 = nx.DiGraph()
    for i in range(n_genes):
        prior_grn_1.add_node(f"gene_{i}")
    # Add regulatory edges for cell type 1
    prior_grn_1.add_edge("gene_4", "gene_3")  # gene_4 regulates gene_3
    prior_grn_1.add_edge("gene_5", "gene_3")  # gene_5 regulates gene_3
    prior_grn_1.add_edge("gene_3", "gene_1")  # gene_3 regulates gene_1
    prior_grn_1.add_edge("gene_1", "gene_4")  # gene_1 regulates gene_4
    prior_grn_1.add_edge("gene_0", "gene_2")  # gene_0 regulates gene_2
    prior_grn_1.add_edge("gene_2", "gene_5")  # gene_2 regulates gene_5
    
    prior_grns = {0: prior_grn_0, 1: prior_grn_1}
    
    return {
        'gene_expression': gene_expression,
        'cell_positions': cell_positions,
        'genes': genes,
        'cell_type_assignments': cell_type_assignments,
        'ligand_receptor_pairs': ligand_receptor_pairs,
        'receptor_gene_pairs': receptor_gene_pairs,
        'prior_grns': prior_grns,
        'n_time_points': n_time_points,
        'n_cells': n_cells,
        'n_genes': n_genes
    }


def create_hex_grid_test_data():
    """Create toy data for testing the GraphConstructor with a hexagonal grid layout"""
    # Define dimensions
    n_time_points = 15
    n_cells = 7  # Center cell plus 6 surrounding cells
    n_genes = 6
    
    # Create gene expression tensor: (n_time_points, n_cells, n_genes)
    gene_expression = torch.zeros((n_time_points, n_cells, n_genes))
    for t in range(n_time_points):
        for c in range(n_cells):
            for g in range(n_genes):
                gene_expression[t, c, g] = t*100 + c*10 + g
    
    # Create spatial positions tensor: (n_time_points, n_cells, 2)
    # Arrange cells in a hexagonal pattern with distance 10 between adjacent cells
    cell_positions = torch.zeros((n_time_points, n_cells, 2))
    
    # Constants for hexagonal layout
    hex_distance = 10.0
    angle_step = 2 * np.pi / 6  # 60 degrees in radians
    
    for t in range(n_time_points):
        # Center cell (index 0)
        cell_positions[t, 0] = torch.tensor([0.0, 0.0])
        
        # Surrounding cells (indices 1-6)
        for i in range(6):
            angle = i * angle_step
            x = hex_distance * np.cos(angle)
            y = hex_distance * np.sin(angle)
            cell_positions[t, i+1] = torch.tensor([x, y])
    
    # Define gene names
    genes = [f"gene_{i}" for i in range(n_genes)]
    
    # Define cell type assignments: alternating pattern
    cell_type_assignments = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long)
    
    # Define ligand-receptor pairs
    ligand_receptor_pairs = [
        ("gene_0", "gene_1"),  # gene_0 is ligand, gene_1 is receptor
        ("gene_2", "gene_3"),  # gene_2 is ligand, gene_3 is receptor
    ]
    
    # Define receptor-gene pairs (selective connections)
    receptor_gene_pairs = [
        ("gene_1", "gene_4"),  # receptor gene_1 regulates gene_4
        ("gene_1", "gene_5"),  # receptor gene_1 also regulates gene_5
        ("gene_3", "gene_1"),  # receptor gene_3 regulates gene_1
        ("gene_3", "gene_4"),  # receptor gene_3 also regulates gene_4
        ("gene_4", "gene_2"),  # gene_4 regulates gene_2
        ("gene_4", "gene_3"),  # gene_4 regulates gene_3
        ("gene_5", "gene_3"),  # gene_5 regulates gene_3
        ("gene_5", "gene_0"),  # gene_5 regulates gene_0
    ]
    
    # Create prior gene regulatory networks (GRNs) for each cell type
    # GRN for cell type 0
    prior_grn_0 = nx.DiGraph()
    for i in range(n_genes):
        prior_grn_0.add_node(f"gene_{i}")
    # Add regulatory edges for cell type 0
    prior_grn_0.add_edge("gene_4", "gene_2")
    prior_grn_0.add_edge("gene_4", "gene_3")
    prior_grn_0.add_edge("gene_5", "gene_3")
    prior_grn_0.add_edge("gene_5", "gene_0")
    prior_grn_0.add_edge("gene_1", "gene_5")
    prior_grn_0.add_edge("gene_2", "gene_4")
    
    # GRN for cell type 1
    prior_grn_1 = nx.DiGraph()
    for i in range(n_genes):
        prior_grn_1.add_node(f"gene_{i}")
    # Add regulatory edges for cell type 1
    prior_grn_1.add_edge("gene_4", "gene_3")
    prior_grn_1.add_edge("gene_5", "gene_3")
    prior_grn_1.add_edge("gene_3", "gene_1")
    prior_grn_1.add_edge("gene_1", "gene_4")
    prior_grn_1.add_edge("gene_0", "gene_2")
    prior_grn_1.add_edge("gene_2", "gene_5")
    
    prior_grns = {0: prior_grn_0, 1: prior_grn_1}
    
    return {
        'gene_expression': gene_expression,
        'cell_positions': cell_positions,
        'genes': genes,
        'cell_type_assignments': cell_type_assignments,
        'ligand_receptor_pairs': ligand_receptor_pairs,
        'receptor_gene_pairs': receptor_gene_pairs,
        'prior_grns': prior_grns,
        'n_time_points': n_time_points,
        'n_cells': n_cells,
        'n_genes': n_genes
    }

# create_test_data = create_square_grid_data
create_test_data = create_hex_grid_test_data

def retrieve_simulated_data(data_dir="data/raw"):
    """
    Load simulated data from the specified directory.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing simulated data files
        
    Returns:
    --------
    dict
        Dictionary containing all simulated data components
    """
    # Create an empty dictionary to store loaded data
    data = {}
    
    # Verify the directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Define the main simulation results file path
    sim_file_path = os.path.join(data_dir, "simulation_results.pkl")

    # Load the simulation results
    with open(sim_file_path, 'rb') as f:
        sim_data = pickle.load(f)
    # Extract data from the loaded simulation results
    # Based on the saving function structure:
    # - 'genes' is a 3D array (time_points x cells x genes)
    # - 'positions' is a 3D array (time_points x cells x 2)
    # - 'metadata' contains time_points, cell_ids, gene_names, cell_types, and prior_grns
    
    # Extract gene expression data (time_points x cells x genes)
    data['gene_expression'] = torch.tensor(sim_data['genes'])
    
    # Extract cell positions (time_points x cells x 2)
    data['cell_positions'] = torch.tensor(sim_data['positions'])
    
    # Extract metadata
    metadata = sim_data['metadata']
    
    # Extract gene names
    data['genes'] = metadata['gene_names']
    
    # Extract cell type assignments
    cell_ids = metadata['cell_ids']
    cell_types_dict = metadata['cell_types']
    
    # Create a mapping from cell IDs to their corresponding types
    unique_cell_types = sorted(set(cell_types_dict.values()))
    label_to_int = {label: idx for idx, label in enumerate(unique_cell_types)}
    print(label_to_int)
    # Map each cell ID to its corresponding integer label
    assignments = [label_to_int[cell_types_dict[cell_id]] for cell_id in cell_ids]

    data['cell_type_assignments'] = torch.tensor(assignments, dtype=torch.long)
    
    # Extract prior GRNs
    cell_specific_prior_grns =  [metadata['prior_grns'][cell_type] for cell_type in label_to_int.keys()]
    data['prior_grns'] = cell_specific_prior_grns

    data['receptor_gene_pairs'] = metadata['receptor_gene_pairs']
    data['ligand_gene_pairs'] = metadata['ligand_gene_pairs']
    data['ligand_receptor_pairs'] = metadata['ligand_receptor_pairs']

     # Calculate dimensions
    data['n_time_points'] = data['gene_expression'].shape[0]
    data['n_cells'] = data['gene_expression'].shape[1]
    data['n_genes'] = data['gene_expression'].shape[2]

    return data

def visualize_graph(graph, title, output_dir='results', save_plot=True, show_plot=False, figsize=(10, 10), return_pos=False):
    """
    Visualize a NetworkX graph with node types shown in different colors
    
    Args:
        graph: NetworkX graph to visualize
        title: Title for the plot
        output_dir: Directory to save the visualization
        save_plot: Whether to save the plot to file (default: True)
        show_plot: Whether to display the plot (default: False)
    """
    plt.figure(figsize=figsize)
    
    # Group nodes by type
    gene_nodes = [n for n, d in graph.nodes(data=True) if d.get('type', 'gene') == 'gene']
    receptor_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'receptor']
    ligand_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'ligand']
    input_ligand_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'input_ligand']
    
    # Create positions for the graph layout
    pos = {}
    
    # Position gene nodes in a circle at the bottom
    gene_pos = nx.circular_layout(nx.Graph([(g, g2) for g in gene_nodes for g2 in gene_nodes if g != g2]))
    gene_pos = {k: (v[0], v[1] - 0.5) for k, v in gene_pos.items()}
    pos.update(gene_pos)
    
    # Position receptor nodes above genes
    receptor_pos = nx.circular_layout(nx.Graph([(r, r2) for r in receptor_nodes for r2 in receptor_nodes if r != r2]))
    receptor_pos = {k: (v[0], v[1] + 0.0) for k, v in receptor_pos.items()}
    pos.update(receptor_pos)
    
    # Position ligand nodes above receptors
    ligand_pos = nx.circular_layout(nx.Graph([(l, l2) for l in ligand_nodes for l2 in ligand_nodes if l != l2]))
    ligand_pos = {k: (v[0], v[1] + 0.5) for k, v in ligand_pos.items()}
    pos.update(ligand_pos)
    
    # Position input ligand nodes to the right
    for i, node in enumerate(input_ligand_nodes):
        # Extract cell and gene info from the node name
        cell = graph.nodes[node].get('cell')
        gene = graph.nodes[node].get('gene')
        # Position based on cell ID (horizontal spread) and gene (vertical position)
        pos[node] = (1.5 + int(cell) * 0.2, 0.0 + int(gene.split('_')[1]) * 0.1)
    
    # Draw the nodes with different colors by type
    nx.draw_networkx_nodes(graph, pos, nodelist=gene_nodes, node_color='lightblue', 
                          node_size=500, label='Genes')
    nx.draw_networkx_nodes(graph, pos, nodelist=receptor_nodes, node_color='lightgreen', 
                          node_size=500, label='Receptors')
    nx.draw_networkx_nodes(graph, pos, nodelist=ligand_nodes, node_color='orange', 
                          node_size=500, label='Ligands')
    nx.draw_networkx_nodes(graph, pos, nodelist=input_ligand_nodes, node_color='salmon', 
                          node_size=500, label='Input Ligands')
    
    # Draw edges with different colors by type
    gene_to_gene_edges = [(u, v) for u, v in graph.edges() 
                         if u in gene_nodes and v in gene_nodes]
    gene_to_ligand_edges = [(u, v) for u, v in graph.edges() 
                           if u in gene_nodes and v in ligand_nodes]
    input_ligand_to_receptor_edges = [(u, v) for u, v in graph.edges() 
                                     if u in input_ligand_nodes and v in receptor_nodes]
    receptor_to_gene_edges = [(u, v) for u, v in graph.edges() 
                             if u in receptor_nodes and v in gene_nodes]
    other_edges = [(u, v) for u, v in graph.edges() 
                  if (u, v) not in gene_to_gene_edges + gene_to_ligand_edges + 
                     input_ligand_to_receptor_edges + receptor_to_gene_edges]
    
    nx.draw_networkx_edges(graph, pos, edgelist=gene_to_gene_edges, 
                          edge_color='blue', width=1.0, alpha=0.7)
    nx.draw_networkx_edges(graph, pos, edgelist=gene_to_ligand_edges, 
                          edge_color='orange', width=1.0, alpha=0.7)
    nx.draw_networkx_edges(graph, pos, edgelist=input_ligand_to_receptor_edges, 
                          edge_color='red', width=1.0, alpha=0.7)
    nx.draw_networkx_edges(graph, pos, edgelist=receptor_to_gene_edges, 
                          edge_color='green', width=1.0, alpha=0.7)
    nx.draw_networkx_edges(graph, pos, edgelist=other_edges, 
                          edge_color='gray', width=1.0, alpha=0.5)
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=8)
    
    # Add legend, title, and other formatting
    plt.title(title)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    
    # Save and/or show the plot
    if save_plot:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        output_file = os.path.join(output_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved graph visualization to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    if return_pos:
        return pos

def visualize_cell_positions(cell_positions, time_point, output_dir='results', save_plot=True, show_plot=False):
    """
    Visualize cell positions at a specific time point
    
    Args:
        cell_positions: Tensor of shape (n_time_points, n_cells, 2)
        time_point: Time point to visualize
        output_dir: Directory to save the visualization
        save_plot: Whether to save the plot to file (default: True)
        show_plot: Whether to display the plot (default: False)
    """
    plt.figure(figsize=(8, 8))
    
    # Extract positions for the given time point
    positions = cell_positions[time_point].numpy()
    
    # Plot each cell
    for i, pos in enumerate(positions):
        plt.scatter(pos[0], pos[1], s=100, label=f"Cell {i}")
        plt.text(pos[0], pos[1], f"Cell {i}", fontsize=12)
    
    # Draw distance circles for reference (radius 10 and 15)
    for i, pos in enumerate(positions):
        circle1 = plt.Circle((pos[0], pos[1]), 10.0, fill=False, linestyle='--', 
                            alpha=0.3, color='gray')
        circle2 = plt.Circle((pos[0], pos[1]), 15.0, fill=False, linestyle=':', 
                            alpha=0.3, color='gray')
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
    
    plt.title(f"Cell Positions at Time Point {time_point}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    
    if save_plot:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        output_file = os.path.join(output_dir, f"Cell_Positions_t{time_point}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved cell positions visualization to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_feature_values(graph, gene_expression, cell_idx, time_point, 
                           delta_gl, delta_lr, delta_rg, delta_gg, node_features,
                           output_dir='results', save_plot=True, show_plot=False):
    """
    Visualize feature values for nodes in the graph
    
    Args:
        graph: NetworkX graph
        gene_expression: Tensor of shape (n_time_points, n_cells, n_genes)
        cell_idx: Cell index
        time_point: Current time point
        delta_gl, delta_lr, delta_rg, delta_gg: Time lags
        node_features: Dictionary mapping nodes to assigned features
        output_dir: Directory to save the visualization
        save_plot: Whether to save the plot to file (default: True)
        show_plot: Whether to display the plot (default: False)
    """
    plt.figure(figsize=(14, 10))
    
    # Create a table of node types, source data, and assigned features
    node_data = []
    for node in graph.nodes():
        node_type = graph.nodes[node].get('type', 'gene')
        
        # Determine expected feature based on node type and time lag
        expected_feature = None
        
        if node_type == 'gene':
            gene_idx = int(node.split('_')[1])
            expr_time = time_point - delta_gg
            if expr_time >= 0:
                expected_feature = gene_expression[expr_time, cell_idx, gene_idx].item()
                source = f"gene_expression[{expr_time}, {cell_idx}, {gene_idx}]"
            
        elif node_type == 'ligand':
            gene = graph.nodes[node]['gene']
            gene_idx = int(gene.split('_')[1])
            expr_time = time_point - delta_gl
            if expr_time >= 0:
                expected_feature = gene_expression[expr_time, cell_idx, gene_idx].item()
                source = f"gene_expression[{expr_time}, {cell_idx}, {gene_idx}]"
            
        elif node_type == 'input_ligand':
            neighbor_cell_idx = graph.nodes[node]['cell']
            gene = graph.nodes[node]['gene']
            gene_idx = int(gene.split('_')[1])
            expr_time = time_point - delta_lr
            if expr_time >= 0:
                expected_feature = gene_expression[expr_time, neighbor_cell_idx, gene_idx].item()
                source = f"gene_expression[{expr_time}, {neighbor_cell_idx}, {gene_idx}]"
            
        elif node_type == 'receptor':
            gene = graph.nodes[node]['gene']
            gene_idx = int(gene.split('_')[1])
            expr_time = time_point - delta_rg
            if expr_time >= 0:
                expected_feature = gene_expression[expr_time, cell_idx, gene_idx].item()
                source = f"gene_expression[{expr_time}, {cell_idx}, {gene_idx}]"
        
        if expected_feature is not None:
            assigned = node_features.get(node, [None])[0]
            node_data.append({
                'Node': node,
                'Type': node_type,
                'Source': source,
                'Expected': f"{expected_feature:.4f}",
                'Assigned': f"{assigned:.4f}" if assigned is not None else "None",
                'Match': "Yes" if expected_feature == assigned else "No"
            })
    
    # Create a figure for the table
    fig, ax = plt.subplots(figsize=(12, len(node_data) * 0.4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(
        cellText=[[d['Node'], d['Type'], d['Source'], d['Expected'], d['Assigned'], d['Match']] 
                  for d in node_data],
        colLabels=['Node', 'Type', 'Source', 'Expected', 'Assigned', 'Match'],
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.1, 0.3, 0.15, 0.15, 0.1]
    )
    
    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Highlight mismatches
    for i, d in enumerate(node_data):
        if d['Match'] == 'No':
            for j in range(6):
                table[(i+1, j)].set_facecolor('salmon')
    
    plt.title(f"Node Feature Validation for Cell {cell_idx} at Time {time_point}")
    plt.tight_layout()
    
    if save_plot:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        output_file = os.path.join(output_dir, f"Feature_Validation_Cell{cell_idx}_t{time_point}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved feature validation to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def validate_node_features(graph, pyg_graph, gene_expression, cell_idx, time_point,
                         delta_gl, delta_lr, delta_rg, delta_gg, gene_indices):
    """
    Validate that node features are correctly assigned according to time lags and neighbor relationships
    
    Args:
        graph: NetworkX graph
        pyg_graph: PyTorch Geometric graph
        gene_expression: Tensor of shape (n_time_points, n_cells, n_genes)
        cell_idx: Cell index
        time_point: Current time point
        delta_gl, delta_lr, delta_rg, delta_gg: Time lags
        gene_indices: Dictionary mapping gene names to indices
        
    Returns:
        is_valid: Boolean indicating whether all features match expected values
        node_features: Dictionary of assigned features
        mismatches: List of mismatched nodes
    """
    node_list = list(graph.nodes())
    is_valid = True
    mismatches = []
    node_features = {}
    
    # Extract features from the PyG graph
    features = pyg_graph.x.numpy()
    
    for i, node in enumerate(node_list):
        node_type = graph.nodes[node].get('type', 'gene')
        assigned_feature = features[i][0]  # Get the feature from PyG graph
        node_features[node] = [assigned_feature]
        
        # Calculate expected feature based on node type and time lag
        expected_feature = None
        
        if node_type == 'gene':
            gene_idx = gene_indices[node]
            expr_time = time_point - delta_gg
            if expr_time >= 0:
                expected_feature = gene_expression[expr_time, cell_idx, gene_idx].item()
            
        elif node_type == 'ligand':
            gene = graph.nodes[node]['gene']
            gene_idx = gene_indices[gene]
            expr_time = time_point - delta_gl
            if expr_time >= 0:
                expected_feature = gene_expression[expr_time, cell_idx, gene_idx].item()
            
        elif node_type == 'input_ligand':
            neighbor_cell_idx = graph.nodes[node]['cell']
            gene = graph.nodes[node]['gene']
            gene_idx = gene_indices[gene]
            expr_time = time_point - delta_lr
            if expr_time >= 0:
                expected_feature = gene_expression[expr_time, neighbor_cell_idx, gene_idx].item()
            
        elif node_type == 'receptor':
            gene = graph.nodes[node]['gene']
            gene_idx = gene_indices[gene]
            expr_time = time_point - delta_rg
            if expr_time >= 0:
                expected_feature = gene_expression[expr_time, cell_idx, gene_idx].item()
        
        # Check if the feature matches the expected value
        if expected_feature is not None:
            if abs(assigned_feature - expected_feature) > 1e-6:
                is_valid = False
                mismatches.append({
                    'node': node,
                    'type': node_type,
                    'expected': expected_feature,
                    'assigned': assigned_feature
                })
    
    return is_valid, node_features, mismatches

def validate_receptor_connections(graph, receptor_gene_pairs):
    """
    Validate that receptor-gene connections in the graph match the specified pairs
    
    Args:
        graph: NetworkX graph
        receptor_gene_pairs: List of (receptor, gene) pairs
        
    Returns:
        is_valid: Boolean indicating whether connections match specifications
        mismatches: List of unexpected or missing connections
    """
    is_valid = True
    mismatches = []
    
    # Create lookup set of valid receptor-gene pairs
    valid_pairs = set()
    for receptor, target in receptor_gene_pairs:
        if receptor.startswith('gene_'):  # Only process receptor genes
            receptor_node = f"r_{receptor}"
            valid_pairs.add((receptor_node, target))
    
    # Check all receptor node connections in graph
    for node in graph.nodes():
        if node.startswith('r_'):  # Found a receptor node
            for target in graph.successors(node):
                # Skip if target is not a gene node (e.g. another receptor)
                if not target.startswith('gene_'):
                    continue
                    
                # Check if this connection is valid
                if (node, target) not in valid_pairs:
                    is_valid = False
                    mismatches.append(f"Unexpected connection: {node} -> {target}")
    
    # Check that valid pairs that could exist do exist
    for receptor_node, target in valid_pairs:
        if receptor_node in graph.nodes() and target in graph.nodes():
            if not graph.has_edge(receptor_node, target):
                is_valid = False
                mismatches.append(f"Missing connection: {receptor_node} -> {target}")
    
    return is_valid, mismatches


def test_graph_constructor():
    """
    Test the GraphConstructor class with toy data and visualize graphs for all cells
    """
    # Create test data
    data = create_test_data()
    
    # Initialize the GraphConstructor with receptor_gene_pairs
    graph_constructor = GraphConstructor(
        genes=data['genes'],
        ligand_receptor_pairs=data['ligand_receptor_pairs'],
        receptor_gene_pairs=data['receptor_gene_pairs'],
        cell_type_assignments=data['cell_type_assignments'],
        prior_grns=data['prior_grns']
    )
    
    # Use a larger time point to accommodate our time lags
    time_point = 10
    
    # Visualize cell positions
    visualize_cell_positions(data['cell_positions'], time_point)
    
    # Define time lags as specified by the user
    delta_gl = 1  # Time lag for gene -> ligand
    delta_lr = 5  # Time lag for ligand -> receptor
    delta_rg = 3  # Time lag for receptor -> gene
    delta_gg = 7  # Time lag for gene -> gene
    
    # Report the time lags being used
    print(f"\nUsing time lags:")
    print(f"  - Gene -> Ligand (delta_gl): {delta_gl}")
    print(f"  - Ligand -> Receptor (delta_lr): {delta_lr}")
    print(f"  - Receptor -> Gene (delta_rg): {delta_rg}")
    print(f"  - Gene -> Gene (delta_gg): {delta_gg}")
    
    # Ensure time_point is large enough to handle the lags
    max_lag = max(delta_gl, delta_lr, delta_rg, delta_gg)
    if time_point < max_lag:
        time_point = max_lag
        print(f"Adjusted time_point to {time_point} to handle time lags")
    
    # Process each cell to create and visualize its graph
    results = {}
    for cell_idx in range(data['n_cells']):
        print(f"\n{'='*50}")
        print(f"TESTING CELL {cell_idx}")
        print(f"{'='*50}")
        
        # Test construct_base_graph
        print(f"\nTesting construct_base_graph for cell {cell_idx}...")
        base_graph = graph_constructor.construct_base_graph(cell_idx)
        
        # Print graph statistics
        print(f"Base graph has {base_graph.number_of_nodes()} nodes and {base_graph.number_of_edges()} edges")
        print("Base graph node types:")
        node_types = {}
        for node, attrs in base_graph.nodes(data=True):
            node_type = attrs.get('type', 'gene')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        for t, count in node_types.items():
            print(f"  - {t}: {count} nodes")
        
        # Visualize the base graph
        visualize_graph(base_graph, f"Base Graph for Cell {cell_idx}")
        
        # Test update_graph_with_neighbors
        print(f"\nTesting update_graph_with_neighbors for cell {cell_idx} at time {time_point}...")
        
        # Try with different distance thresholds to see the effect
        for distance_threshold in [10.0, 15.0]:
            updated_graph = graph_constructor.update_graph_with_neighbors(
                base_graph, cell_idx, data['cell_positions'], time_point,
                distance_threshold=distance_threshold
            )
            
            # Print graph statistics
            print(f"\nWith distance threshold {distance_threshold}:")
            print(f"Updated graph has {updated_graph.number_of_nodes()} nodes and {updated_graph.number_of_edges()} edges")
            print("Updated graph node types:")
            node_types = {}
            for node, attrs in updated_graph.nodes(data=True):
                node_type = attrs.get('type', 'gene')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            for t, count in node_types.items():
                print(f"  - {t}: {count} nodes")
            
            # Visualize the updated graph
            visualize_graph(updated_graph, f"Updated Graph for Cell {cell_idx} (d={distance_threshold})")
        
        # Test assign_node_features with specified time lags
        print(f"\nTesting assign_node_features for cell {cell_idx} at time {time_point}...")
        
        # Use distance threshold 15.0 to include diagonal neighbors
        updated_graph = graph_constructor.update_graph_with_neighbors(
            base_graph, cell_idx, data['cell_positions'], time_point,
            distance_threshold=15.0
        )
        
        # Assign node features
        pyg_graph = graph_constructor.assign_node_features(
            updated_graph, cell_idx, time_point, data['gene_expression'],
            delta_gl, delta_lr, delta_rg, delta_gg
        )
        
        # Print PyTorch Geometric graph information
        print(f"\nPyTorch Geometric graph:")
        print(f"Number of nodes: {pyg_graph.num_nodes}")
        print(f"Number of edges: {pyg_graph.num_edges}")
        print(f"Node features shape: {pyg_graph.x.shape}")
        print(f"Gene node indices: {pyg_graph.gene_node_indices}")
        
        # Validate node features
        is_valid, node_features, mismatches = validate_node_features(
            updated_graph, pyg_graph, data['gene_expression'], cell_idx, time_point,
            delta_gl, delta_lr, delta_rg, delta_gg, graph_constructor.gene_indices
        )
        
        # Validate receptor connections
        print("\nValidating receptor-gene connections...")
        is_valid_receptor, mismatches_receptor = validate_receptor_connections(updated_graph, data['receptor_gene_pairs'])
        if is_valid_receptor:
            print("✅ Receptor connections are correct")
        else:
            print("❌ Found issues with receptor connections:")
            for mismatch in mismatches_receptor:
                print(f"  - {mismatch}")
        
        # Visualize feature values for validation
        visualize_feature_values(
            updated_graph, data['gene_expression'], cell_idx, time_point,
            delta_gl, delta_lr, delta_rg, delta_gg, node_features
        )
        
        # Print validation results
        if is_valid:
            print(f"\n✅ Feature validation passed for Cell {cell_idx}! All node features match expected values.")
        else:
            print(f"\n❌ Feature validation failed for Cell {cell_idx}! Found mismatches:")
            for mismatch in mismatches:
                print(f"  - Node {mismatch['node']} ({mismatch['type']}): "
                      f"Expected {mismatch['expected']:.4f}, got {mismatch['assigned']:.4f}")
        
        # Store results for this cell
        results[cell_idx] = {
            'base_graph': base_graph,
            'updated_graph': updated_graph,
            'pyg_graph': pyg_graph,
            'feature_validation': {
                'is_valid': is_valid,
                'node_features': node_features,
                'mismatches': mismatches
            }
        }
    
    # Summary of validation results for all cells
    print("\n" + "="*50)
    print("VALIDATION SUMMARY FOR ALL CELLS")
    print("="*50)
    all_valid = True
    for cell_idx in range(data['n_cells']):
        is_valid = results[cell_idx]['feature_validation']['is_valid']
        all_valid = all_valid and is_valid
        status = "✅ PASSED" if is_valid else "❌ FAILED"
        print(f"Cell {cell_idx}: {status}")
    
    # Final summary
    if all_valid:
        print("\n✅ All tests completed successfully for all cells!")
    else:
        print("\n❌ Tests completed with feature validation errors for some cells.")
    
    print("\nCheck the results directory for visualizations.")
    
    return {
        'data': data,
        'cell_results': results
    }


def print_gene_expression_table(gene_expression, n_time_points, n_cells, n_genes):
    """
    Print a table of gene expression values for better understanding the test data
    
    Args:
        gene_expression: Tensor of shape (n_time_points, n_cells, n_genes)
        n_time_points: Number of time points
        n_cells: Number of cells
        n_genes: Number of genes
    """
    print("\nGene Expression Data:")
    for t in range(n_time_points):
        print(f"\nTime point {t}:")
        print("Cell | " + " | ".join([f"Gene {g}" for g in range(n_genes)]))
        print("-" * (6 + n_genes * 10))
        for c in range(n_cells):
            # Get the actual values for this cell at this time point
            values = [gene_expression[t, c, g].item() for g in range(n_genes)]
            print(f"{c:4d} | " + " | ".join([f"{val:7.2f}" for val in values]))


def visualize_all_cells_comparison(data, output_dir='results', save_plot=True, show_plot=False):
    """
    Create a comparative visualization of all cells' positions and their neighborhoods
    
    Args:
        data: Test data dictionary
        output_dir: Directory to save the visualization
        save_plot: Whether to save the plot to file (default: True)
        show_plot: Whether to display the plot (default: False)
    """
    time_point = 10  # Use the same time point as in the test
    
    plt.figure(figsize=(12, 10))
    
    # Extract positions for the given time point
    positions = data['cell_positions'][time_point].numpy()
    
    # Plot each cell
    for i, pos in enumerate(positions):
        plt.scatter(pos[0], pos[1], s=200, label=f"Cell {i}")
        plt.text(pos[0] + 0.5, pos[1] + 0.5, f"Cell {i}", fontsize=14)
    
    # Draw connections between cells within threshold distance
    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions):
            if i != j:
                distance = np.linalg.norm(pos_i - pos_j)
                if distance <= 15.0:  # Using 15.0 as our larger threshold
                    # Draw a line between cells
                    plt.plot([pos_i[0], pos_j[0]], [pos_i[1], pos_j[1]], 
                            'k--', alpha=0.3, linewidth=1)
                    # Annotate with distance
                    mid_x = (pos_i[0] + pos_j[0]) / 2
                    mid_y = (pos_i[1] + pos_j[1]) / 2
                    plt.text(mid_x, mid_y, f"{distance:.1f}", 
                            fontsize=10, ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.7))
    
    # Draw distance circles for reference
    for i, pos in enumerate(positions):
        circle = plt.Circle((pos[0], pos[1]), 15.0, fill=False, linestyle='-', 
                          alpha=0.3, color='green')
        plt.gca().add_patch(circle)
    
    plt.title(f"Cell Positions and Connections at Time Point {time_point}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal')
    
    if save_plot:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        output_file = os.path.join(output_dir, f"All_Cells_Comparison_t{time_point}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved all cells comparison to {output_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    # Create test data
    # data = create_test_data()
    
    data = retrieve_simulated_data()
    # Only print a subset of the time points to keep output readable
    print_subset_time_points = [0, 3, 5, 7, 10, 14]  # Selected time points to print
    
    print("\nGene Expression Data (selected time points):")
    for t in print_subset_time_points:
        print(f"\nTime point {t}:")
        print("Cell | " + " | ".join([f"Gene {g}" for g in range(data['n_genes'])]))
        print("-" * (6 + data['n_genes'] * 10))
        for c in range(data['n_cells']):
            # Get the actual values for this cell at this time point
            values = [data['gene_expression'][t, c, g].item() for g in range(data['n_genes'])]
            print(f"{c:4d} | " + " | ".join([f"{val:7.2f}" for val in values]))
    
    # Create comparative visualization of all cells
    visualize_all_cells_comparison(data)
    
    # Run the tests
    results = test_graph_constructor()
    
    print("\nCheck the results directory for visualizations.") 