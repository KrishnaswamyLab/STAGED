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

class SpatialTemporalDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
    
    @property
    def raw_file_names(self):
        """List the raw files that must be found in the raw directory."""
        return ["data.pt"]  # Change based on your actual raw files

    @property
    def processed_file_names(self):
        """List of files that should be in the processed directory after processing."""
        return ["graph_{}.pt".format(i) for i in range(len(os.listdir(self.processed_dir)))]

    def download(self):
        """Download raw data if necessary. Override this method to handle downloading."""
        pass  # Implement if needed

    def process(self):
        """Process raw data into graph objects and save them."""
        
        ##TODO correct this processing step
        raw_data = torch.load(os.path.join(self.raw_dir, "data.pt"))  # Example raw data

        (time, num_cells, 2) = spatial_data.shape 
        (time, num_cells, genes) = raw_data.shape 
        (num_cell_types, graph_priors) = prior_graphs_data.shape
        (num_cells,1) = cell_types.shape # cell types label for each cell on the dataset

        # Initial GRN
        ## We retrieve the GRN relative to that cell type.
        GRN = prior_graphs_data[cell_type]
        # Add output ligand and input receptors
        GRN_ligand_receptors = add_ligand_receptors(spatial_data)

        # Connect the GRN to its neighbors
        ## Connect the neighbors receptors with the ligands based on spatial locations
        neighbors - find_neighbors(spatial_data, cell_types)
        GRN_neighbors = connect_neighbors(neighbors, GRN_ligand_receptors)
        GRN_final = GRN_ligand_receptors + GRN_neighbors

        # Assign node features to the GRN
        ## Assign the GRN nodes with the genes expression levels based on the time lag
        GRN_final[node_features] = retrieve_genes_expression(neighbors,GRN_final, time_lag)
        
        prior_GRN_features = GRN.node_features
        time_laged_GRN_features = GRN_final.time_lagged_features

        # Assign node features using time lag

        # for i, graph in enumerate(raw_data):
        #     data = Data(
        #         x=graph["x"], 
        #         edge_index=graph["edge_index"], 
        #         y=graph["y"]
        #     )

        torch.save(data, os.path.join(self.processed_dir, f"graph_{i}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Load a processed graph"""
        return torch.load(os.path.join(self.processed_dir, f"graph_{idx}.pt"))

if __name__ == "__main__":
    dataset = SpatialTemporalDataset(root='data')
    dataset.process()