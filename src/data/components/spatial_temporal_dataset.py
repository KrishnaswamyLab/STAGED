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

        self.time_lag = time_lag
        self.num_neighbors = num_neighbors
        self.prior_graphs = None

        print(f"Initialized with time_lag: {self.time_lag}, num_neighbors: {self.num_neighbors}")
        super().__init__(root, transform, pre_transform)


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
        gene_trajectories = torch.load(os.path.join(self.raw_dir, "raw_data.pt"))  # (time, num_cells, genes)
        spatial_time_series = torch.load(os.path.join(self.raw_dir, "spatial_data.pt"))  # (time, num_cells, 2)
        cell_types = torch.load(os.path.join(self.raw_dir, "cell_types.pt"))  # (num_cells, 1)

        # Load Prior graphs
        prior_graphs = torch.load(os.path.join(self.raw_dir, "prior_graphs.pt"))  # (num_cell_types, graph_priors)
        self.prior_graphs = prior_graphs

        num_timepoints, num_cells, num_genes = gene_trajectories.shape
        # num_cell_types, _ = prior_graphs.shape

        print('Loaded all data')
        # Iterate through timepoints to create dynamic graphs
        for t in range(self.time_lag, num_timepoints):
            cell_type_specific_grns = []
            
            for cell_idx in range(num_cells):

                cell_type = cell_types[cell_idx].item()

                # Retrieve the current cell GRN
                cell_GRN = self.retrieveCellGRN(cell_type,t)

                # Find neighboring cells and connect ligand-receptor interactions
                neighbors = self.find_neighbor_cells(cell_idx, self.num_neighbors, spatial_time_series[t])

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
        # Convert adjacency matrix to edge_index
        edge_index = torch.nonzero(prior_graph, as_tuple=True)

        # Node features (optional: initialize with zeros or relevant features)
        num_nodes = prior_graph.size(0)  # Assuming square adjacency matrix
        x = torch.zeros((num_nodes, 1))  # Placeholder for node features

        # Create PyG Data object
        ##TODO: THIS IS WRONG!
        prior_graph = Data(x=x, edge_index=edge_index)
        
        #TODO: Retrieve the CellGRN based on time points.

        return prior_graph
        # # Convert adjacency matrix to edge_index if needed
        # edge_index = torch.nonzero(prior_graph, as_tuple=True)
        
        # # Node features (optional: initialize with zeros, ones, or relevant features)
        # num_nodes = edge_index.max().item() + 1  # Assuming nodes are indexed from 0
        
        # x = torch.zeros((num_nodes, self.num_gene_features))  # Placeholder

        # return Data(x=x, edge_index=edge_index)

    
    def find_neighbor_cells(self, cell_idx: int, num_neighbors: int, spatial_data: torch.Tensor):
        """
        Find the `num_neighbors` closest cells to a specific cell based on spatial coordinates.

        Args:
            cell_idx (int): Index of the cell for which to find neighbors.
            num_neighbors (int): Number of nearest neighbors to find.
            spatial_data (torch.Tensor): Tensor of shape (num_cells, 2) containing x, y coordinates.

        Returns:
            List[int]: A list containing the indices of `num_neighbors` nearest neighbors.
        """
        # Convert spatial data to numpy for KDTree
        tree = KDTree(spatial_data.cpu().numpy())  # Faster neighbor search
        
        # Find the `num_neighbors + 1` closest neighbors (including self)
        _, neighbor_indices = tree.query(spatial_data[cell_idx].cpu().numpy(), k=num_neighbors + 1)
        
        # Remove self from neighbors (first element is the cell itself)
        neighbor_indices = neighbor_indices[1:]
        
        return neighbor_indices.tolist()  # Convert to list
    
    def add_neighbors_ligand_receptors_pairs(self, cell_GRN: Data, neighbors: list):
        """
        Connect ligand-receptor interactions between the given cell's GRN and its neighbors.

        Args:
            cell_GRN (Data): PyG Data object representing the current cell's gene regulatory network (GRN).
            neighbors (list): List of neighbor cell indices.
            spatial_data (torch.Tensor): (num_cells, 2) Tensor of spatial coordinates.

        Returns:
            Data: Updated PyG Data object with ligand-receptor connections added.
        """

        edge_index = cell_GRN.edge_index # Copy existing edges
        num_nodes = cell_GRN.x.shape[0]  # Number of nodes in the current cell's GRN

        new_edges = []  # Store new ligand-receptor edges
        for neighbor_idx in neighbors:
            # Example: Create an edge between ligand (node 0) and receptor (node 1) of the neighbor
            ligand_idx = 0  # Replace with actual ligand node index
            receptor_idx = 1  # Replace with actual receptor node index

            # Adjust receptor index based on neighbor's node indexing
            neighbor_receptor_idx = receptor_idx + (neighbor_idx * num_nodes)

            new_edges.append([ligand_idx, neighbor_receptor_idx])  # Ligand → Neighbor Receptor

        if new_edges:
            import pdb;
            pdb.set_trace()
            new_edges = torch.tensor(new_edges, dtype=torch.long).T  # Convert to PyG format (2, num_edges)
            edge_index = torch.cat([edge_index, new_edges], dim=1)  # Append new edges

        return Data(x=cell_GRN.x, edge_index=edge_index)

    def retrieve_genes_expression_environment(self, GRN_final, gene_expression, neighbors):
        # Implement logic to retrieve gene expression for the environment
        # This is a placeholder function
        return gene_expression
    

  # Extract cell IDs and time points

        cell_ids = list(gene_expression_data.keys())
        
        # Determine all available time points
        all_time_points = set()
        for cell_id in cell_ids:
            for gene_idx in range(self.num_genes):
                if gene_idx in gene_expression_data[cell_id]:
                    all_time_points.update(gene_expression_data[cell_id][gene_idx].keys())
        time_points = sorted(all_time_points)
        
        # Calculate total time range
        total_time_steps = len(time_points)
        
        # Set train_end_time if not provided
        if train_end_time is None:
            train_end_time = int(0.7 * total_time_steps)  # Use 70% for training by default
        
        # Separate train and test time points
        train_time_points = [t for t in time_points if t < train_end_time]
        test_time_points = [t for t in time_points if t >= train_end_time]
        
        # Calculate initial time steps needed
        t_init = self.model.get_t_init()
        
        # Split training time points into train and validation sets
        # but only use time points after t_init for predictable points
        predictable_train_time_points = [t for t in train_time_points if t > t_init]
        
        if len(predictable_train_time_points) > 0:
            num_val_points = max(1, int(validation_fraction * len(predictable_train_time_points)))
            # Use the latest time points in the training set for validation
            val_time_points = predictable_train_time_points[-num_val_points:]
            # Use the remaining time points for training
            train_time_points_for_loss = [t for t in predictable_train_time_points if t not in val_time_points]
        else:
            # If no predictable time points, we can't do validation
            train_time_points_for_loss = predictable_train_time_points
            val_time_points = []
        
        print(f"Time-based split: Training on time points {train_time_points}")
        print(f"                  Validation on time points {val_time_points}")
        print(f"                  Testing on time points {test_time_points}")

if __name__ == "__main__":
    dataset = SpatialTemporalDataset(root='data',time_lag=1,num_neighbors=3)
    dataset.process()