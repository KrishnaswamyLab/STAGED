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
        return ["traj_gene_sp.pt"]  # Change based on your actual raw files

    @property
    def processed_file_names(self):
        """List of files that should be in the processed directory after processing."""
        return ["graph_{}.pt".format(i) for i in range(len(os.listdir(self.processed_dir)))]

    def download(self):
        """Download raw data if necessary. Override this method to handle downloading."""
        pass  # Implement if needed

    def process(self):
        """Process raw data into graph objects and save them."""
        raw_data = torch.load(os.path.join(self.raw_dir, "data.pt"))  # Example raw data
        (time, num_genes, num_cells_per_type) = raw_data.shape 
        print(raw_data.shape) #should delete
        # for i, graph in enumerate(raw_data):
        #     data = Data(
        #         x=graph["x"], 
        #         edge_index=graph["edge_index"], 
        #         y=graph["y"]
        #     )
        #     torch.save(data, os.path.join(self.processed_dir, f"graph_{i}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """Load a processed graph"""
        return torch.load(os.path.join(self.processed_dir, f"graph_{idx}.pt"))

if __name__ == "__main__":
    dataset = SpatialTemporalDataset(root='data')