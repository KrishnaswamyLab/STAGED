import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader


class GraphDataHandler:
    """
    Handles the conversion between individual graphs and batched data,
    and processes data through the STAGED model.
    """
    def __init__(self, model, device=None):
        """
        Initialize the data handler with the model
        
        Args:
            model: STAGED model instance
            device: Device to use for computation (defaults to CUDA if available)
        """
        self.model = model
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.model = self.model.to(self.device)

    def process_cell_graphs(self, cell_graphs, num_genes, batch_size=None):
        """
        Process a list of cell graphs through the STAGED model
        
        Args:
            cell_graphs: List of PyTorch Geometric Data objects for each cell
            num_genes: Number of genes in the model
            batch_size: Batch size for processing (None = process all at once)
            
        Returns:
            predictions: Tensor of shape (n_cells, n_genes) with predicted values
            attention_weights: List of attention weight tensors
            node_ptr: Tensor containing cumulative node counts for each graph
        """
        if not cell_graphs:
            raise ValueError("Empty graph list provided")
        
        if batch_size is not None and batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        # Move all graphs to the device
        for i in range(len(cell_graphs)):
            cell_graphs[i] = cell_graphs[i].to(self.device)
        
        # Initialize predictions tensor
        n_cells = len(cell_graphs)
        predictions = torch.zeros(n_cells, num_genes, device=self.device)
        all_attention_weights = []
        
        # Process graphs based on batch size
        if batch_size is None or batch_size >= len(cell_graphs):
            # Process all graphs in one batch
            batch_data = Batch.from_data_list(cell_graphs)
            batch_data = batch_data.to(self.device)
            
            # Get node embeddings and attention weights
            node_embeddings, attention = self.model(batch_data)
            all_attention_weights = [attention]  # Single attention tuple
            
            # Get all gene indices at once
            batch_gene_indices = []
            batch_cell_indices = []
            local_gene_positions = []
            
            for cell_idx, graph in enumerate(cell_graphs):
                batch_mask = batch_data.batch == cell_idx
                graph_node_indices = torch.where(batch_mask)[0]
                gene_indices_in_batch = graph_node_indices[graph.gene_node_indices]
                
                batch_gene_indices.append(gene_indices_in_batch)
                batch_cell_indices.extend([cell_idx] * len(gene_indices_in_batch))
                local_gene_positions.extend(range(len(gene_indices_in_batch)))
            
            # Concatenate all gene indices
            all_gene_indices = torch.cat(batch_gene_indices)
            
            # Get predictions for all genes at once
            all_gene_preds = self.model.predict_genes(node_embeddings, all_gene_indices)
            
            # Assign predictions to the correct positions
            for pred_idx, (cell_idx, local_gene_idx) in enumerate(zip(batch_cell_indices, local_gene_positions)):
                predictions[cell_idx, local_gene_idx] = all_gene_preds[pred_idx]
            
        else:
            # Use DataLoader for batch processing
            loader = DataLoader(cell_graphs, batch_size=batch_size, shuffle=False)
            
            start_idx = 0
            for batch in loader:
                batch = batch.to(self.device)
                
                # Process the batch
                node_embeddings, attention = self.model(batch)
                all_attention_weights.append(attention)
                
                # Gather all gene indices for the batch at once
                batch_gene_indices = []
                batch_cell_indices = []
                local_gene_positions = []
                
                batch_size_actual = min(batch_size, len(cell_graphs) - start_idx)
                for i in range(batch_size_actual):
                    cell_idx = start_idx + i
                    
                    batch_mask = batch.batch == i
                    graph_node_indices = torch.where(batch_mask)[0]
                    gene_indices_in_batch = graph_node_indices[cell_graphs[cell_idx].gene_node_indices]
                    
                    batch_gene_indices.append(gene_indices_in_batch)
                    batch_cell_indices.extend([cell_idx] * len(gene_indices_in_batch))
                    local_gene_positions.extend(range(len(gene_indices_in_batch)))
                
                # Concatenate all gene indices
                all_gene_indices = torch.cat(batch_gene_indices)
                
                # Get predictions for all genes in the batch at once
                all_gene_preds = self.model.predict_genes(node_embeddings, all_gene_indices)
                
                # Assign predictions to the correct positions
                for pred_idx, (cell_idx, local_gene_idx) in enumerate(zip(batch_cell_indices, local_gene_positions)):
                    predictions[cell_idx, local_gene_idx] = all_gene_preds[pred_idx]
                
                start_idx += batch_size
        
        # Concatenate attention weights before returning
        concatenated_attention = self.concatenate_attention(all_attention_weights)
        
        # Store cumulative node counts for each graph
        node_counts = [g.num_nodes for g in cell_graphs]
        node_ptr = torch.tensor([0] + list(torch.cumsum(torch.tensor(node_counts), dim=0)))
        
        return predictions, concatenated_attention, node_ptr

    def concatenate_attention(self, attention_list):
        """
        Concatenate attention weights from multiple batches
        
        Args:
            attention_list: List of (edge_index, attention_values) tuples
            
        Returns:
            tuple: (concatenated_edge_indices, concatenated_attention_values)
        """
        edge_indices = []
        attention_values = []
        offset = 0
        
        for edge_index, values in attention_list:
            # Adjust edge indices by the offset for proper concatenation
            adjusted_edge_index = edge_index.clone()
            adjusted_edge_index += offset
            
            edge_indices.append(adjusted_edge_index)
            attention_values.append(values)
            
            # Update offset based on maximum node index
            offset = max(adjusted_edge_index.max() + 1, offset)
        
        return (torch.cat(edge_indices, dim=1), torch.cat(attention_values, dim=0))

    def split_attention_by_graphs(self, attention_weights, node_ptr):
        """
        Split concatenated attention weights back into per-graph attention weights
        
        Args:
            attention_weights: Tuple of (edge_index, attention_values)
            node_ptr: Tensor containing cumulative node counts for each graph
            
        Returns:
            list: List of (edge_index, attention_values) tuples for each graph
        """
        edge_index, attn_values = attention_weights
        num_graphs = len(node_ptr) - 1
        graph_attentions = []
        
        for i in range(num_graphs):
            start_idx = node_ptr[i]
            end_idx = node_ptr[i + 1]
            
            # Find edges that belong to this graph
            edge_mask = (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)
            graph_edges = edge_index[:, edge_mask] - start_idx
            graph_attn = attn_values[edge_mask]
            
            graph_attentions.append((graph_edges, graph_attn))
        
        return graph_attentions 