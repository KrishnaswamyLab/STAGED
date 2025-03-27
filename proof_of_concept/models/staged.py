import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
import networkx as nx
import numpy as np


class STAGED(nn.Module):
    """
    STAGED (Spatiotemporal Analysis of Gene Expression Dynamics) model
    Implements a graph-based model for predicting gene expression trajectories
    with spatial and temporal context.
    """
    def __init__(
        self,
        num_genes,
        hidden_dim=64,
        num_gat_layers=1,
        num_mlp_layers=2,
        dropout=0.1,
        delta_gl=1,  # Time lag for gene -> ligand
        delta_lr=1,  # Time lag for ligand -> receptor 
        delta_rg=1,  # Time lag for receptor -> gene
        delta_gg=1,  # Time lag for gene -> gene
    ):
        super(STAGED, self).__init__()
        
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.delta_gl = delta_gl
        self.delta_lr = delta_lr
        self.delta_rg = delta_rg
        self.delta_gg = delta_gg
        
        # Initial feature dimensions
        self.input_dim = 1  # Single gene expression value
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(self.input_dim, hidden_dim, heads=1, dropout=dropout))
        
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        
        # MLP for prediction
        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        for _ in range(num_mlp_layers - 2):
            self.mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.mlp_layers.append(nn.Linear(hidden_dim, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, cell_graphs, gene_expression_history, cell_positions):
        """
        Forward pass of the STAGED model
        
        Args:
            cell_graphs: List of cell-specific graphs (one per cell)
            gene_expression_history: Dictionary of gene expression histories
            cell_positions: Spatial positions of cells
            
        Returns:
            predicted_expression: Predicted gene expression values
            attention_weights: Attention weights from GAT layers
        """
        predictions = {}
        attention_weights = {}
        
        # Process each cell's graph
        for cell_idx, graph in enumerate(cell_graphs):
            # Apply GAT layers
            x = graph.x
            edge_index = graph.edge_index
            
            for gat_layer in self.gat_layers:
                x, attention = gat_layer(x, edge_index, return_attention_weights=True)
                x = self.relu(x)
                x = self.dropout(x)
            
            # Apply MLP for final prediction
            node_embeddings = x
            
            gene_predictions = {}
            # Extract gene nodes (not ligand or receptor nodes)
            gene_nodes = graph.gene_node_indices
            
            for gene_idx, node_idx in enumerate(gene_nodes):
                gene_embedding = node_embeddings[node_idx]
                
                # Pass through MLP
                x_mlp = gene_embedding
                for mlp_layer in self.mlp_layers[:-1]:
                    x_mlp = mlp_layer(x_mlp)
                    x_mlp = self.relu(x_mlp)
                    x_mlp = self.dropout(x_mlp)
                
                # Final prediction
                gene_prediction = self.mlp_layers[-1](x_mlp)
                # Reshape to match expected dimensions [1, 1] instead of [1]
                gene_prediction = gene_prediction.view(1, 1)
                gene_predictions[gene_idx] = gene_prediction
            
            predictions[cell_idx] = gene_predictions
            attention_weights[cell_idx] = attention
        
        return predictions, attention_weights

    def get_t_init(self):
        """Return the initial time steps needed before prediction can start"""
        return max(self.delta_gl, self.delta_lr, self.delta_rg, self.delta_gg) 