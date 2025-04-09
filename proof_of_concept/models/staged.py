import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch


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
        delta_lr=5,  # Time lag for ligand -> receptor 
        delta_rg=3,  # Time lag for receptor -> gene
        delta_gg=7,  # Time lag for gene -> gene
        add_self_loops=True,
    ):
        super(STAGED, self).__init__()
        
        self.num_genes = num_genes
        self.hidden_dim = hidden_dim
        self.delta_gl = delta_gl
        self.delta_lr = delta_lr
        self.delta_rg = delta_rg
        self.delta_gg = delta_gg
        self.add_self_loops = add_self_loops
        
        # Initial feature dimensions
        self.input_dim = 1  # Single gene expression value
        
        # GAT layers
        assert num_gat_layers == 1, "Must have exactly one GAT layer"
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(self.input_dim, hidden_dim, heads=1, dropout=dropout, add_self_loops=add_self_loops)) # optionally use GATv2Conv? TODO investigate difference.
        
        for _ in range(num_gat_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, add_self_loops=add_self_loops))
        
        # MLP for prediction using Sequential
        mlp_layers = []
        mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_mlp_layers - 2):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
        
        mlp_layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*mlp_layers)
    
    def forward(self, batch_data):
        """
        Forward pass of the STAGED model
        
        Args:
            batch_data: PyTorch Geometric Data or Batch object
                Can be a single graph or a batch of graphs
            
        Returns:
            node_embeddings: Node embeddings after GAT layers
            attention_weights: Attention weights from the last GAT layer
        """
        x = batch_data.x
        edge_index = batch_data.edge_index
        
        # Track attention weights from the last layer
        attention = None
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            # The GATConv automatically respects graph boundaries in batched data
            x, attention = gat_layer(x, edge_index, return_attention_weights=True) # only returns the last layer's attention weights
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        return x, attention
    
    def predict_genes(self, node_embeddings, gene_indices):
        """
        Generate predictions for gene nodes
        
        Args:
            node_embeddings: Embeddings for all nodes
            gene_indices: Indices of the gene nodes to predict
            
        Returns:
            predictions: Gene expression predictions [num_genes, 1]
        """
        # Get embeddings for gene nodes only
        gene_embeddings = node_embeddings[gene_indices]
        
        # Apply MLP to get predictions
        predictions = self.mlp(gene_embeddings)
        
        return predictions
        
    def get_t_init(self):
        """Return the initial time steps needed before prediction can start"""
        return max(self.delta_gl, self.delta_lr, self.delta_rg, self.delta_gg) 