import torch
from torch_geometric.data import Batch
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
        """
        # Move all graphs to the device
        for i in range(len(cell_graphs)):
            cell_graphs[i] = cell_graphs[i].to(self.device)
        
        # Initialize predictions tensor
        n_cells = len(cell_graphs)
        predictions = torch.zeros(n_cells, num_genes, device=self.device)
        all_attention_weights = []
        
        # Determine how to process the graphs
        if batch_size is None or batch_size >= len(cell_graphs):
            # Process all graphs in one batch
            if len(cell_graphs) > 1:
                batch_data = Batch.from_data_list(cell_graphs)
                batch_data = batch_data.to(self.device)
                
                # Get node embeddings and attention weights
                node_embeddings, attention = self.model(batch_data)
                
                # Store attention weights
                all_attention_weights.append(attention)
                
                # Process each cell's results
                for cell_idx, graph in enumerate(cell_graphs):
                    # Get the node indices for this graph in the batch
                    batch_mask = batch_data.batch == cell_idx
                    graph_node_indices = torch.where(batch_mask)[0]
                    
                    # Map gene indices in the original graph to the batch
                    gene_indices_in_batch = graph_node_indices[graph.gene_node_indices]
                    
                    # Get predictions for this cell's genes
                    gene_preds = self.model.predict_genes(node_embeddings, gene_indices_in_batch)
                    
                    # Add to the predictions tensor
                    for local_gene_idx, global_gene_idx in enumerate(graph.gene_node_indices):
                        gene_idx = local_gene_idx  # In our setup, local_gene_idx maps to actual gene index
                        predictions[cell_idx, gene_idx] = gene_preds[local_gene_idx]
            else:
                # Single graph case
                graph = cell_graphs[0]
                node_embeddings, attention = self.model(graph)
                
                # Store attention weights
                all_attention_weights.append(attention)
                
                # Get predictions for gene nodes
                gene_preds = self.model.predict_genes(node_embeddings, graph.gene_node_indices)
                
                # Add to the predictions tensor
                for local_gene_idx, node_idx in enumerate(graph.gene_node_indices):
                    gene_idx = local_gene_idx  # In our setup, local_gene_idx maps to actual gene index
                    predictions[0, gene_idx] = gene_preds[local_gene_idx]
        else:
            # Use DataLoader for batch processing
            loader = DataLoader(cell_graphs, batch_size=batch_size, shuffle=False)
            
            start_idx = 0
            for batch in loader:
                batch = batch.to(self.device)
                
                # Process the batch
                node_embeddings, attention = self.model(batch)
                
                # Store attention weights
                all_attention_weights.append(attention)
                
                # Process each cell in the batch
                for i in range(min(batch_size, len(cell_graphs) - start_idx)):
                    cell_idx = start_idx + i
                    
                    # Get the node indices for this graph in the batch
                    batch_mask = batch.batch == i
                    graph_node_indices = torch.where(batch_mask)[0]
                    
                    # Map gene indices in the original graph to the batch
                    graph = cell_graphs[cell_idx]
                    gene_indices_in_batch = graph_node_indices[graph.gene_node_indices]
                    
                    # Get predictions for this cell's genes
                    gene_preds = self.model.predict_genes(node_embeddings, gene_indices_in_batch)
                    
                    # Add to the predictions tensor
                    for local_gene_idx, node_idx in enumerate(graph.gene_node_indices):
                        gene_idx = local_gene_idx  # In our setup, local_gene_idx maps to actual gene index
                        predictions[cell_idx, gene_idx] = gene_preds[local_gene_idx]
                
                start_idx += batch_size
                
        return predictions, all_attention_weights 