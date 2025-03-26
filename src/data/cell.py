import torch
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt


class CellNetwork:
    def __init__(self, num_cells=4, nodes_per_cell=4):
        """
        Initialize a cell network model.
        
        Parameters:
        -----------
        num_cells : int
            Number of cells in the network
        nodes_per_cell : int
            Number of nodes per cell (typically 4: G1, G2, G3, L, R)
        """
        self.num_cells = num_cells
        self.nodes_per_cell = nodes_per_cell
        self.total_nodes = num_cells * nodes_per_cell
        
        # Node indices within each cell
        self.G1 = 0  # Gene 1
        self.G2 = 1  # Gene 2
        self.G3 = 2  # Ligand gene
        self.L = 3   # Ligand
        self.R = 4   # Receptor
        
        # Cell positions (default: 2x2 grid)
        self.cell_coords = self.initialize_cell_grid()
        self.distances = self.calculate_distances()
        
    def initialize_cell_grid(self, grid_size=None):
        """
        Initialize cells in a grid pattern.
        
        Parameters:
        -----------
        grid_size : tuple, optional
            Grid dimensions (rows, cols). If None, will create a square-like grid.
            
        Returns:
        --------
        torch.Tensor
            Cell coordinates as a tensor of shape [num_cells, 2]
        """
        if grid_size is None:
            # Determine grid dimensions to be approximately square
            side = int(np.ceil(np.sqrt(self.num_cells)))
            grid_size = (side, side)
            
        coords = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                if len(coords) < self.num_cells:
                    coords.append([i, j])
        
        return torch.tensor(coords, dtype=torch.float)
    
    def initialize_random_positions(self, bounds=(0, 10)):
        """
        Initialize cells with random positions.
        
        Parameters:
        -----------
        bounds : tuple
            Min and max coordinates for random positioning
            
        Returns:
        --------
        torch.Tensor
            Cell coordinates as a tensor of shape [num_cells, 2]
        """
        low, high = bounds
        self.cell_coords = torch.rand(self.num_cells, 2) * (high - low) + low
        self.distances = self.calculate_distances()
        return self.cell_coords
        
    def calculate_distances(self):
        """Calculate pairwise distances between all cells"""
        return torch.cdist(self.cell_coords, self.cell_coords, p=2)
    
    def create_graph(self, gene_expression=None):
        """
        Create the graph structure for the cell network.
        
        Parameters:
        -----------
        gene_expression : torch.Tensor, optional
            Initial gene expression values. If None, random values are generated.
            
        Returns:
        --------
        Data
            PyTorch Geometric Data object containing the graph
        """
        # Initialize gene expression
        if gene_expression is None:
            x = torch.randn(self.total_nodes, 1).clamp(min=0.0)
        else:
            x = gene_expression
            
        edge_index = []
        
        # Add intra-cellular connections
        for c in range(self.num_cells):
            base = c * self.nodes_per_cell
            
            # Connect genes within cell
            for src in [self.G1, self.G2, self.G3]:
                for tgt in [self.G1, self.G2, self.G3]:
                    edge_index.append([base + src, base + tgt])
                    
            # Connect ligand to genes (only G3 is connected to L)
            edge_index.append([base + self.G3, base + self.L])
            
            # Connect receptor to genes
            for gene in [self.G1, self.G2, self.G3]:
                edge_index.append([base + self.R, base + gene])
        
        # Add inter-cellular connections (ligand -> receptor)
        for c in range(self.num_cells):
            r_idx = c * self.nodes_per_cell + self.R
            for cp in range(self.num_cells):
                if cp != c:
                    l_idx = cp * self.nodes_per_cell + self.L
                    edge_index.append([l_idx, r_idx])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    
    def plot_network(self, data=None, node_size=300, figsize=(10, 8)):
        """Plot the cell network with spatial positioning"""
        if data is None:
            data = self.create_graph()
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot cells
        ax.scatter(self.cell_coords[:, 0], self.cell_coords[:, 1], 
                    s=node_size*2, alpha=0.3, color='gray')
        
        # Label cells
        for i, (x, y) in enumerate(self.cell_coords):
            ax.text(x, y, f"Cell {i}", ha='center', va='center')
        
        # Plot intercellular edges (ligand -> receptor)
        for edge_idx in range(data.edge_index.shape[1]):
            src, tgt = data.edge_index[:, edge_idx]
            src_cell = src.item() // self.nodes_per_cell
            tgt_cell = tgt.item() // self.nodes_per_cell
            src_type = src.item() % self.nodes_per_cell
            tgt_type = tgt.item() % self.nodes_per_cell
            
            if src_cell != tgt_cell and src_type == self.L and tgt_type == self.R:
                src_x, src_y = self.cell_coords[src_cell]
                tgt_x, tgt_y = self.cell_coords[tgt_cell]
                ax.arrow(src_x, src_y, tgt_x-src_x, tgt_y-src_y, 
                        head_width=0.1, head_length=0.1, fc='blue', ec='blue',
                        length_includes_head=True, alpha=0.5)
                
        # create a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                        markersize=10, label='G1, G2, G3 (Genes)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                        markersize=10, label='L (Ligand)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                        markersize=10, label='R (Receptor)'),
            plt.Line2D([0], [0], color='blue', lw=2, alpha=0.5, label='Ligand → Receptor')
        ]
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                    fancybox=True, shadow=True, ncol=2)
        plt.title("Cell Network")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.7)
        return fig, ax