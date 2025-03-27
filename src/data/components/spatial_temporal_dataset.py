from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

import torch_geometric.data 
import os
import numpy as np

import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

from scipy.spatial import KDTree

class SpatialTemporalDataset(Dataset):
    def __init__(self, root, time_lag, num_neighbors, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.time_lag = time_lag
        self.num_neighbors = num_neighbors
        self.prior_graphs = None

    @property
    def raw_file_names(self):
        """List the raw files that must be found in the raw directory."""
        return [
            "data.pt",  # Gene trajectories
            "spatial_data.pt",  # Spatial coordinates
            "prior_graphs.pt",  # Prior graph information
            "cell_types.pt"  # Cell type labels
        ]

    @property
    def processed_file_names(self):
        """List of files that should be in the processed directory after processing."""
        return ["graph_{}.pt".format(i) for i in range(len(os.listdir(self.processed_dir)))]

    def download(self):
        """Download raw data if necessary. Override this method to handle downloading."""
        pass  # Implement if needed

    def process(self):
        """Process raw spatial and gene expression data into graph objects and save them."""
        
        # Load raw data
        gene_trajectories = torch.load(os.path.join(self.raw_dir, "data.pt"))  # (time, num_cells, genes)
        spatial_time_series = torch.load(os.path.join(self.raw_dir, "spatial_data.pt"))  # (time, num_cells, 2)
        cell_types = torch.load(os.path.join(self.raw_dir, "cell_types.pt"))  # (num_cells, 1)

        # Load Prior graphs
        prior_graphs = torch.load(os.path.join(self.raw_dir, "prior_graphs.pt"))  # (num_cell_types, graph_priors)
        self.prior_graphs = prior_graphs

        num_timepoints, num_cells, num_genes = gene_trajectories.shape
        num_cell_types, _ = prior_graphs.shape

        # Iterate through timepoints to create dynamic graphs
        for t in range(self.time_lag, num_timepoints):
            cell_type_specific_grns = []
            
            for cell_idx in range(num_cells):

                cell_type = cell_types[cell_idx].item()

                # Retrieve the current cell GRN
                cell_GRN = self.retrieveCellGRN(cell_type,t)

                # Find neighboring cells and connect ligand-receptor interactions
                neighbors = self.find_neighbor_cells(self.num_neighbors, spatial_time_series[t], cell_types)

                # Add ligand-receptor interactions based on current time_point and spatial data
                final_GRN = self.add_neighbors_ligand_receptors_pairs(cell_GRN, neighbors)
                
                # Assign node features using time-lagged gene expression
                h_c_hat = self.assign_node_features(final_GRN, gene_trajectories,neighbors)
                
                
                # Convert to PyG Data object
                graph_data = Data(
                    x=h_c_hat,  # (num_nodes, num_features)
                    edge_index= final_GRN.edge_index,  # (2, num_edges)
                    y=gene_trajectories[t]  # Prediction target could be full expression profile
                )

                cell_type_specific_grns.append(graph_data)

            # Save processed graphs
            for i, graph in enumerate(cell_type_specific_grns):
                torch.save(graph, os.path.join(self.processed_dir, f"graph_t{t}_c{i}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Load a processed graph"""
        return torch.load(os.path.join(self.processed_dir, f"graph_{idx}.pt"))
    
    def assign_node_features(final_GRN, gene_trajectories,neighbors):
        # Implement logic to assign node features based on gene expression and neighbors
        # This is a placeholder function
        node_features = torch.zeros((final_GRN.num_nodes, gene_trajectories.shape[2]))
        for i in range(len(neighbors)):
            node_features[i] = gene_trajectories[neighbors[i]]
        return node_features
    
    def retrieveCellGRN(self, cell_type: int, t: int) -> Data:
        """
        Retrieve the gene regulatory network (GRN) for a specific cell type.
        Args:
            cell_type (int): The index of the cell type.
        
        Returns:
            Data: A PyG Data object representing the GRN.
        """
        # Retrieve the adjacency matrix or edge list for this cell type
        prior_graph = self.prior_graphs[cell_type]  # (num_nodes, num_nodes) or (2, num_edges)
        
        #TODO: Retrieve the CellGRN based on time points.

        return prior_graph
        # # Convert adjacency matrix to edge_index if needed
        # edge_index = torch.nonzero(prior_graph, as_tuple=True)
        
        # # Node features (optional: initialize with zeros, ones, or relevant features)
        # num_nodes = edge_index.max().item() + 1  # Assuming nodes are indexed from 0
        
        # x = torch.zeros((num_nodes, self.num_gene_features))  # Placeholder

        # return Data(x=x, edge_index=edge_index)

    
    def find_neighbor_cells(num_neighbors: int, spatial_data: torch.Tensor, cell_types: torch.Tensor):
        """
        Find the `num_neighbors` closest cells for each cell based on spatial coordinates.

        Args:
            num_neighbors (int): Number of nearest neighbors to find.
            spatial_data (torch.Tensor): Tensor of shape (num_cells, 2) containing x, y coordinates.
            cell_types (torch.Tensor): Tensor of shape (num_cells, 1) containing cell type labels.

        Returns:
            List[List[int]]: A list where each index `i` contains the indices of `num_neighbors` nearest neighbors.
        """
        num_cells = spatial_data.shape[0]
        
        # Convert spatial data to numpy for KDTree
        tree = KDTree(spatial_data.cpu().numpy())  # Faster neighbor search
        
        # Find the `num_neighbors + 1` closest neighbors (including self)
        _, neighbor_indices = tree.query(spatial_data.cpu().numpy(), k=num_neighbors + 1) 
        
        # Remove self from neighbors (first element in each row is the cell itself)
        neighbor_indices = neighbor_indices[:, 1:]
        
        return neighbor_indices.tolist()  # Convert to list of lists
    
    def add_neighbors_ligand_receptors_pairs(cell_GRN: Data, neighbors: list):
        """
        Connect ligand-receptor interactions between the given cell's GRN and its neighbors.

        Args:
            cell_GRN (Data): PyG Data object representing the current cell's gene regulatory network (GRN).
            neighbors (list): List of neighbor cell indices.
            spatial_data (torch.Tensor): (num_cells, 2) Tensor of spatial coordinates.

        Returns:
            Data: Updated PyG Data object with ligand-receptor connections added.
        """

        edge_index = cell_GRN.edge_index.clone()  # Copy existing edges
        num_nodes = cell_GRN.x.shape[0]  # Number of nodes in the current cell's GRN

        new_edges = []  # Store new ligand-receptor edges

        for neighbor in neighbors:
            print(neighbor)
            # Example: Create an edge between ligand (node 0) and receptor (node 1) of the neighbor
            ligand_idx = 0  # Replace with actual ligand node index
            receptor_idx = 1  # Replace with actual receptor node index

            # Adjust receptor index based on neighbor's node indexing
            neighbor_receptor_idx = receptor_idx + (neighbor * num_nodes)

            new_edges.append([ligand_idx, neighbor_receptor_idx])  # Ligand → Neighbor Receptor

        if new_edges:
            new_edges = torch.tensor(new_edges, dtype=torch.long).T  # Convert to PyG format (2, num_edges)
            edge_index = torch.cat([edge_index, new_edges], dim=1)  # Append new edges

        return Data(x=cell_GRN.x, edge_index=edge_index)

    def retrieve_genes_expression_environment(GRN_final, gene_expression, neighbors):
        # Implement logic to retrieve gene expression for the environment
        # This is a placeholder function
        return gene_expression
    
if __name__ == "__main__":
    dataset = SpatialTemporalDataset(root='data',time_lag=1,num_neighbors=3)
    dataset.process()