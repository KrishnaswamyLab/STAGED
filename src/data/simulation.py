import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image
import sys
import os
sys.path.append('/gpfs/gibbs/pi/krishnaswamy_smita/kx44/projects/STAGED')
from src.data.cell import CellNetwork

class CellSimulation:
    def __init__(self, num_cells=4, nodes_per_cell=5, delta1 = 1, delta2=2, delta3=3, deltal = 5):
        """
        Simulation of cellular gene regulation network with explicit time stepping
        
        Parameters:
        -----------
        num_cells : int
            Number of cells in the simulation
        nodes_per_cell : int
            Number of nodes per cell (typically 5: G1, G2, G3, L, R)
        delta1 : int
            Time delay for gene to gene signaling
        delta2 : int
            Time delay for receptor to gene signaling
        delta3 : int
            Time delay for ligand to receptor expression
        deltal: int
            Time delay for receptor recieving ligand signal
        """
        self.network = CellNetwork(num_cells, nodes_per_cell)
        self.data = self.network.create_graph()
        self.num_cells = num_cells
        self.nodes_per_cell = nodes_per_cell
        
        # Node types (matching CellNetwork)
        self.G1 = 0  # Gene 1
        self.G2 = 1  # Gene 2
        self.G3 = 2  # Gene 3
        self.L = 3   # Ligand
        self.R = 4   # Receptor
        
        self.genes = [self.G1, self.G2, self.G3]  # All gene indices
        
        # Time delays
        self.delta1 = delta1  # Delay between genes
        self.delta2 = delta2  # Delay from receptor to gene
        self.delta3 = delta3  # Delay from ligand to receptor
        self.deltal = deltal  # Delay from receptor to ligand
        
        # Initialize cell update functions
        self.cell_updaters = [self.default_gene_updater for _ in range(num_cells)]
        self.ligand_functions = [self.default_ligand_function for _ in range(num_cells)]
        
        # Initial expression
        self.x = torch.zeros(num_cells * nodes_per_cell, 1)
        
    def default_gene_updater(self, input_genes, receptor_input):
        """Default function to update gene expression based on inputs"""
        # Average of input genes plus receptor input
        return 0.7 * sum(input_genes) / len(input_genes) + 0.3 * receptor_input
    
    def default_ligand_function(self, gene_values):
        """Default function for ligand production based on gene expression"""
        # Average of gene expressions, with emphasis on G3
        return 0.4 * gene_values[0] + 0.3 * gene_values[1] + 0.3 * gene_values[2]
        
    def set_updater_function(self, cell_idx, function):
        """Set custom gene update function for a specific cell"""
        self.cell_updaters[cell_idx] = function
    
    def set_ligand_function(self, cell_idx, function):
        """Set custom ligand production function for a specific cell"""
        self.ligand_functions[cell_idx] = function
        
    def initialize_expression(self, initial_values=None):
        """Set initial gene expression values"""
        if initial_values is None:
            # Default: activate first gene in first cell
            self.x = torch.zeros(self.num_cells * self.nodes_per_cell, 1)
            self.x[0] = 1.0  # G1 in first cell
        else:
            self.x = initial_values
    
    def spatial_decay(self, ligand_values, distances, decay_rate=0.5):
        """
        Calculate receptor inputs based on ligand expression and spatial distance
        
        Parameters:
        -----------
        ligand_values : torch.Tensor
            Ligand expression values for each cell
        distances : torch.Tensor
            Distance matrix between cells
        decay_rate : float
            Rate of signal decay with distance
            
        Returns:
        --------
        torch.Tensor
            Receptor input values for each cell
        """
        receptor_inputs = torch.zeros(self.num_cells, 1)
        
        for receiver in range(self.num_cells):
            for sender in range(self.num_cells):
                if sender != receiver:
                    # Calculate signal decay based on distance
                    distance = distances[sender, receiver]
                    decay_factor = torch.exp(-decay_rate * distance)
                    
                    # Add weighted contribution to the receptor input
                    receptor_inputs[receiver] += ligand_values[sender] * decay_factor
                    
        return receptor_inputs
    
    def run_simulation(self, steps=100):
        """Run the simulation for multiple steps"""
        # Store history of expression values
        history = [self.x.clone()]
        
        # Run simulation for specified number of steps
        for t in range(1, steps):
            prev_x = history[t - 1].clone()
            new_x = torch.zeros_like(prev_x)
            
            # Calculate ligand production for each cell based on genes
            l_vals = []
            for c in range(self.num_cells):
                base = c * self.nodes_per_cell
                
                # Get gene expressions for this cell from appropriate time step
                if t - self.delta3 >= 0:
                    gene_values = [history[t - self.delta3][base + gene] for gene in self.genes]
                else:
                    gene_values = [torch.tensor([0.0]) for _ in self.genes]
                
                # Apply ligand function to produce ligand
                l_val = self.ligand_functions[c](gene_values)
                l_vals.append(l_val)
            
            l_vals = torch.stack(l_vals).view(-1, 1)
            
            # Calculate receptor inputs using spatial decay, add time lag deltal
            if t - self.deltal >= 0:
                past_l_vals = []
                for c in range(self.num_cells):
                    base = c * self.nodes_per_cell
                    l_idx = base + self.L
                    past_l_vals.append(history[t - self.deltal][l_idx])
                    
                past_l_vals = torch.stack(past_l_vals).view(-1, 1)
                r_inputs = self.spatial_decay(past_l_vals, self.network.distances)
            else:
                r_inputs = torch.zeros(self.num_cells, 1)
            
            # Update each node in the network
            for c in range(self.num_cells):
                base = c * self.nodes_per_cell
                updater = self.cell_updaters[c]
                
                # Update receptor
                r_idx = base + self.R
                new_x[r_idx] = r_inputs[c]
                
                # Update ligand based on values calculated earlier
                l_idx = base + self.L
                new_x[l_idx] = l_vals[c]
                
                # Update genes based on other genes and receptor input
                for gene in self.genes:
                    g_idx = base + gene
                    # Get inputs from other genes
                    if t - self.delta1 >= 0:
                        input_genes = [history[t - self.delta1][base + g] for g in self.genes]
                    else:
                        input_genes = [torch.tensor([0.0]) for _ in self.genes]
                    # Get receptor input with delay
                    receptor_input = history[t - self.delta2][base + self.R] if t - self.delta2 >= 0 else torch.tensor([0.0])
                    # Update gene expression
                    new_x[g_idx] = updater(input_genes, receptor_input)
            
            # Store the new state
            history.append(new_x)
            
        # Return the full history as a stacked tensor
        return torch.stack(history)
    
    def visualize_network(self, history, time_step=0, figsize=(10, 8)):
        """
        Visualize the network at a specific time step
        
        Parameters:
        -----------
        history : torch.Tensor
            History of expression values [steps, nodes, 1]
        time_step : int
            Time step to visualize
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig, ax
            Figure and axis objects
        """
        # Convert to networkx graph for visualization
        G = to_networkx(self.data, to_undirected=False)
        
        # Create layout for the graph
        pos = nx.spring_layout(G, seed=42)
        
        # Create labels, node shapes, and colors
        labels = {}
        node_shapes = {'gene': [], 'ligand': [], 'receptor': []}
        node_colors = []
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        solid_edges, dashed_edges = [], []
        
        # Set up node properties
        for c in range(self.num_cells):
            base = c * self.nodes_per_cell
            for i in range(self.nodes_per_cell):
                nid = base + i
                if i == self.R:
                    node_shapes['receptor'].append(nid)
                elif i == self.L:
                    node_shapes['ligand'].append(nid)
                else:
                    node_shapes['gene'].append(nid)
                node_colors.append(colors[c % len(colors)])
                
                # Create node label
                if i == self.R:
                    label = f"C{c}_R"
                elif i == self.L:
                    label = f"C{c}_L"
                else:
                    label = f"C{c}_G{i+1}"
                labels[nid] = label
        
        # Set up edge properties
        for u, v in G.edges():
            if u // self.nodes_per_cell == v // self.nodes_per_cell:
                solid_edges.append((u, v))
            else:
                dashed_edges.append((u, v))
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Draw the network
        self.draw_graph(G, history[time_step], time_step, pos, labels, 
                        solid_edges, dashed_edges, node_shapes, node_colors, ax)
        
        return fig, ax
    
    def draw_graph(self, G, node_expr, time, pos, labels, solid_edges, dashed_edges, node_shapes, node_colors, ax):
        """
        Draw the network graph at a specific time step
        
        Parameters:
        -----------
        G : networkx.Graph
            Network graph
        node_expr : torch.Tensor
            Node expression values [nodes, 1]
        time : int
            Time step
        pos : dict
            Node positions
        labels : dict
            Node labels
        solid_edges : list
            Edges to draw with solid lines (intracellular)
        dashed_edges : list
            Edges to draw with dashed lines (intercellular)
        node_shapes : dict
            Dictionary mapping node types to node IDs
        node_colors : list
            Color for each node
        ax : matplotlib.axes.Axes
            Axes to draw on
        """
        ax.clear()
        fixed_size = 1000
        node_expr_flat = node_expr.view(-1).numpy()
        expr_labels = {i: f"{node_expr_flat[i]:.2f}" for i in range(len(node_expr_flat))}
    
        # Draw gene nodes as circles
        nx.draw_networkx_nodes(G, pos,
                                nodelist=node_shapes['gene'],
                                node_color=[node_colors[n] for n in node_shapes['gene']],
                                node_shape='o',
                                node_size=fixed_size,
                                edgecolors='black',
                                alpha=0.7,
                                ax=ax)
        
        # Draw ligand nodes as triangles
        nx.draw_networkx_nodes(G, pos,
                                nodelist=node_shapes['ligand'],
                                node_color=[node_colors[n] for n in node_shapes['ligand']],
                                node_shape='^',
                                node_size=fixed_size,
                                edgecolors='black',
                                alpha=0.7,
                                ax=ax)
    
        # Draw receptor nodes as squares
        nx.draw_networkx_nodes(G, pos,
                                nodelist=node_shapes['receptor'],
                                node_color=[node_colors[n] for n in node_shapes['receptor']],
                                node_shape='s',
                                node_size=fixed_size,
                                edgecolors='black',
                                alpha=0.7,
                                ax=ax)
    
        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges, style='solid', width=2, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, style='dashed', width=2, ax=ax)
        
        # Draw labels with expression values
        nx.draw_networkx_labels(G, pos, labels=expr_labels, font_size=9, ax=ax)
    
        ax.set_title(f"Time Step {time}", fontsize=14)
        ax.axis('off')
        
    def animate_network(self, history=None, steps=100, output_path='gene_simulation.gif', 
                        fps=1, figsize=(10, 8), display=True):
        """
        Create and save an animation of the network dynamics
        
        Parameters:
        -----------
        history : torch.Tensor or None
            History of expression values [steps, nodes, 1]
        steps : int
            Number of steps to simulate if history is None
        output_path : str
            Path to save the output GIF
        fps : int
            Frames per second for the GIF
        figsize : tuple
            Figure size
        display : bool
            Whether to display the animation in the notebook
            
        Returns:
        --------
        IPython.display.Image or None
            The animation if display=True, otherwise None
        """
        # Run simulation if history not provided
        if history is None:
            history = self.run_simulation(steps=steps)
        
        # Convert to networkx graph for visualization
        G = to_networkx(self.data, to_undirected=False)
        
        # Create layout for the graph
        pos = nx.spring_layout(G, seed=42)
        
        # Create labels, node shapes, and colors
        labels = {}
        node_shapes = {'gene': [], 'ligand': [], 'receptor': []}
        node_colors = []
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        solid_edges, dashed_edges = [], []
        
        # Set up node properties
        for c in range(self.num_cells):
            base = c * self.nodes_per_cell
            for i in range(self.nodes_per_cell):
                nid = base + i
                if i == self.R:
                    node_shapes['receptor'].append(nid)
                elif i == self.L:
                    node_shapes['ligand'].append(nid)
                else:
                    node_shapes['gene'].append(nid)
                node_colors.append(colors[c % len(colors)])
                
                # Create node label
                if i == self.R:
                    label = f"C{c}_R"
                elif i == self.L:
                    label = f"C{c}_L"
                else:
                    label = f"C{c}_G{i+1}"
                labels[nid] = label
        
        # Set up edge properties
        for u, v in G.edges():
            if u // self.nodes_per_cell == v // self.nodes_per_cell:
                solid_edges.append((u, v))
            else:
                dashed_edges.append((u, v))
        
        # Create figure for animation
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create update function for animation
        def update(t):
            self.draw_graph(G, history[t], t, pos, labels, 
                            solid_edges, dashed_edges, node_shapes, node_colors, ax)
            return []
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=len(history), interval=1000//fps)
        
        # Save as GIF
        ani.save(output_path, writer=PillowWriter(fps=fps))
        
        # Close figure to prevent display
        plt.close(fig)
        
        print(f"Animation saved to {output_path}")
        
        # Display animation if requested
        if display:
            return Image(filename=output_path)
        return None
    
    def plot_expression_over_time(self, history=None, cells=None, gene_idx=None, 
                                steps=100, figsize=(10, 6), title=None, 
                                legend_loc='best', xlabel='Time Steps', ylabel='Expression Level'):
        """
        Plot gene expression over time for specific cells and genes
    
        Parameters:
        -----------
        history : torch.Tensor or None
            History of expression values [steps, nodes, 1]
        cells : list or None
            List of cell indices to plot. If None, plots all cells
        gene_idx : int or list or None
            Gene index(es) to plot. If None, plots all genes
            0=G1, 1=G2, 2=G3, 3=L, 4=R
        steps : int
            Number of steps to simulate if history is None
        figsize : tuple
            Figure size for the plot
        title : str or None
            Plot title. If None, a default title is generated
        legend_loc : str
            Location for the legend
        xlabel : str
            Label for x-axis
        ylabel : str
            Label for y-axis
        
        Returns:
        --------
        fig, ax
            Figure and axis objects
        """
        # Run simulation if history not provided
        if history is None:
            history = self.run_simulation(steps=steps)
    
        # Determine which cells to plot
        if cells is None:
            cells = list(range(self.num_cells))
        elif isinstance(cells, int):
            cells = [cells]
        
        # Determine which genes to plot
        gene_names = ["Gene 1", "Gene 2", "Gene 3", "Ligand", "Receptor"]
        if gene_idx is None:
            gene_indices = list(range(self.nodes_per_cell))
        elif isinstance(gene_idx, int):
            gene_indices = [gene_idx]
        else:
            gene_indices = gene_idx
    
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
    
        # Line styles and markers for different genes
        line_styles = ['-', '--', '-.', ':', '-']
        markers = ['o', 's', '^', 'D', 'x']
    
        # Color map for different cells
        import matplotlib.cm as cm
        colors = cm.tab10(np.linspace(0, 1, len(cells)))
    
        # Plot the expression over time
        for c_idx, cell in enumerate(cells):
            for g_idx, gene in enumerate(gene_indices):
                node_idx = cell * self.nodes_per_cell + gene
                expression = history[:, node_idx, 0].detach().numpy()
            
                label = f"Cell {cell} - {gene_names[gene]}"
                ax.plot(expression, 
                        label=label,
                        color=colors[c_idx],
                        linestyle=line_styles[g_idx % len(line_styles)],
                        marker=markers[g_idx % len(markers)],
                        markersize=4,
                        markevery=max(1, len(history)//20))
    
        # Set plot title and labels
        if title is None:
            if len(gene_indices) == 1:
                title = f"{gene_names[gene_indices[0]]} Expression Over Time"
            else:
                title = "Gene Expression Over Time"
    
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.7)
    
        # Add legend
        if len(cells) * len(gene_indices) <= 10:  # Only add legend if not too many lines
            ax.legend(loc=legend_loc)
    
        plt.tight_layout()
        return fig, ax
    
    # def save_simulation_results():
    #     #TODO save all data in the format:
    #     # (time, num_cells, 2) = spatial_data.shape 
    #     # (time, num_cells, genes) = raw_data.shape 
    #     # (num_cell_types, graph_priors) = prior_graphs_data.shape
    #     # (num_cells,1) or (num_cells,num_cell_types)  = cell_types.shape # cell types label for each cell on the dataset
    #     return
    
    def save_simulation_results(self, history=None, steps=100, output_dir="./simulation_data", 
                                cell_types=None, num_cell_types=1):
        """
        Save simulation results in standard format for later use
        
        Parameters:
        -----------
        history : torch.Tensor or None
            History of expression values [steps, nodes, 1]
        steps : int
            Number of steps to simulate if history is None
        output_dir : str
            Directory to save the output files
        cell_types : torch.Tensor or None
            Cell type assignments. If None, all cells are assigned to type 0
        num_cell_types : int
            Number of cell types in the simulation
        """
        # Run simulation if history not provided
        if history is None:
            history = self.run_simulation(steps=steps)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Prepare spatial data: (time, num_cells, 2)
        # Use cell coordinates from the network, repeated for each time step
        spatial_data = torch.zeros(len(history), self.num_cells, 2)
        for t in range(len(history)):
            spatial_data[t] = self.network.cell_coords
        
        # 2. Prepare raw data: (time, num_cells, nodes_per_cell)
        # Include all node types (genes, ligand, receptor)
        raw_data = torch.zeros(len(history), self.num_cells, self.nodes_per_cell)
        
        for t in range(len(history)):
            for c in range(self.num_cells):
                for n_idx in range(self.nodes_per_cell):
                    node_idx = c * self.nodes_per_cell + n_idx
                    raw_data[t, c, n_idx] = history[t, node_idx, 0]
        
        # 3. Prepare prior graphs: (num_cell_types, num_nodes, num_nodes)
        # Create an adjacency matrix for each cell type
        edge_index = self.data.edge_index
        total_nodes = self.num_cells * self.nodes_per_cell
        
        # Create adjacency matrix from edge_index
        adj_matrix = torch.zeros(total_nodes, total_nodes)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            adj_matrix[src, dst] = 1.0
        
        # Stack the same graph for each cell type (in a real scenario, might have different graphs)
        prior_graphs = torch.stack([adj_matrix for _ in range(num_cell_types)])
        
        # 4. Prepare cell types: (num_cells, 1) or (num_cells, num_cell_types)
        if cell_types is None:
            # Default: all cells are type 0
            cell_types = torch.zeros(self.num_cells, 1, dtype=torch.long)
        
        # Save the data
        torch.save(spatial_data, os.path.join(output_dir, "spatial_data.pt"))
        torch.save(raw_data, os.path.join(output_dir, "raw_data.pt"))
        torch.save(prior_graphs, os.path.join(output_dir, "prior_graphs.pt"))
        torch.save(cell_types, os.path.join(output_dir, "cell_types.pt"))
        
        print(f"Simulation data saved to {output_dir}:")
        print(f"- Spatial data: {spatial_data.shape}")
        print(f"- Raw data: {raw_data.shape}")
        print(f"- Prior graphs: {prior_graphs.shape}")
        print(f"- Cell types: {cell_types.shape}")
        
        return spatial_data, raw_data, prior_graphs, cell_types