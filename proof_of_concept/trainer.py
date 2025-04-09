import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import networkx as nx

from models.staged import STAGED
from utils.graph_constructor import GraphConstructor
from utils.graph_data_handler import GraphDataHandler


class STAGEDTrainer:
    """
    Trainer class for the STAGED algorithm.
    Handles the full training procedure including graph construction, model training,
    and gene expression prediction.
    """
    def __init__(
        self, 
        genes,
        ligand_receptor_pairs,
        cell_type_assignments,
        prior_grns,
        delta_gl=1,
        delta_lr=5,
        delta_rg=3,
        delta_gg=7,
        hidden_dim=64,
        num_gat_layers=1,
        num_mlp_layers=2,
        learning_rate=0.001,
        weight_decay=1e-5,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the STAGED trainer
        
        Args:
            genes: List of gene identifiers
            ligand_receptor_pairs: List of (ligand, receptor) gene pairs
            cell_type_assignments: Dictionary mapping cell IDs to cell types
            prior_grns: Dictionary mapping cell types to prior GRNs
            delta_gl, delta_lr, delta_rg, delta_gg: Time lags for different connection types
            hidden_dim: Hidden dimension for the model
            num_gat_layers: Number of GAT layers
            num_mlp_layers: Number of MLP layers
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
            device: Device to run the model on
        """
        self.genes = genes
        self.num_genes = len(genes)
        self.ligand_receptor_pairs = ligand_receptor_pairs
        self.cell_type_assignments = cell_type_assignments
        self.prior_grns = prior_grns
        
        # Time lags
        self.delta_gl = delta_gl
        self.delta_lr = delta_lr
        self.delta_rg = delta_rg
        self.delta_gg = delta_gg
        
        # Create the model
        self.model = STAGED(
            num_genes=self.num_genes,
            hidden_dim=hidden_dim,
            num_gat_layers=num_gat_layers,
            num_mlp_layers=num_mlp_layers,
            delta_gl=delta_gl,
            delta_lr=delta_lr,
            delta_rg=delta_rg,
            delta_gg=delta_gg
        ).to(device)
        
        # Create the graph constructor
        self.graph_constructor = GraphConstructor(
            genes=genes,
            ligand_receptor_pairs=ligand_receptor_pairs,
            cell_type_assignments=cell_type_assignments,
            prior_grns=prior_grns
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Device
        self.device = device
        
    def train(
        self,
        gene_expression_data,
        cell_positions,
        train_end_time=None,
        num_epochs=100,
        batch_size=32,
        validation_fraction=0.2,
        patience=10,
        distance_threshold=10.0
    ):
        """
        Train the STAGED model with time-based split
        
        Args:
            gene_expression_data: Tensor of shape (n_time_points, n_cells, n_genes)
                Gene expression data for all cells and time points
            cell_positions: Tensor of shape (n_time_points, n_cells, 2)
                Spatial positions of all cells at all time points
            train_end_time: Time point to end training data (use later time points for testing)
            num_epochs: Number of epochs to train for
            batch_size: Batch size
            validation_fraction: Fraction of training time points to use for validation
            patience: Number of epochs to wait for improvement before early stopping
            distance_threshold: Maximum distance to consider cells as neighbors
            
        Returns:
            losses: Dictionary of training and validation losses
            predictions: Dictionary of predicted gene expression values
        """
        # Extract dimensions from the tensor
        n_time_points, n_cells, n_genes = gene_expression_data.shape
        
        # Determine all available time points (just the indices now)
        time_points = list(range(n_time_points))
        
        # Set train_end_time if not provided
        if train_end_time is None:
            train_end_time = int(0.7 * n_time_points)  # Use 70% for training by default
        
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
        
        # Initialize training state
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        # Initialize gene expression history tensor for model predictions
        # This will be the same shape as the input but will be filled with predictions
        gene_expression_history = gene_expression_data.clone()
        
        # Use all cell indices for training
        train_cells = list(range(n_cells))
        
        # Initialize model and data handler
        model = STAGED(
            num_genes=self.num_genes,
            delta_gl=self.delta_gl,
            delta_lr=self.delta_lr,
            delta_rg=self.delta_rg,
            delta_gg=self.delta_gg
        )
        model = model.to(self.device)

        data_handler = GraphDataHandler(model, device=self.device)

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            # Construct cell-specific graphs for this time point
            cell_graphs = []
            for cell_idx in range(n_cells):
                # Create base graph
                base_graph = self.graph_constructor.construct_base_graph(cell_idx)
                
                # Update with neighbor information
                updated_graph = self.graph_constructor.update_graph_with_neighbors(
                    base_graph, cell_idx, cell_positions, epoch,
                    distance_threshold=distance_threshold
                )
                
                # Assign node features with time lags
                graph_data = self.graph_constructor.assign_node_features(
                    updated_graph, cell_idx, epoch, gene_expression_history,
                    self.delta_gl, self.delta_lr, self.delta_rg, self.delta_gg
                )
                
                cell_graphs.append(graph_data)
            
            # Process all graphs and get tensor predictions
            predictions, _ = data_handler.process_cell_graphs(cell_graphs, num_genes=self.num_genes, batch_size=batch_size)
            
            # Now predictions is a tensor of shape (n_cells, n_genes)
            # You can add it to your gene_expression_history at the right time point
            # For example:
            gene_expression_history[epoch] = predictions
            
            # Calculate loss against next time point ground truth
            next_time_point = epoch + 1
            if next_time_point < n_time_points:
                target = gene_expression_data[next_time_point]  # Shape: (n_cells, n_genes)
                loss = self.criterion(predictions, target)
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            epoch_loss = loss.item()
            train_losses.append(epoch_loss)
            
            # Validation on validation time points (all cells)
            val_loss = 0.0
            num_val_samples = 0
            
            if val_time_points:
                for t in val_time_points:
                    # Validation processing... (similar code structure to the training loop)
                    # ...
                    # (truncated for brevity, would follow same pattern)
            
            # Calculate average validation loss
            val_loss = val_loss / num_val_samples if num_val_samples > 0 else float('inf')
            val_losses.append(val_loss)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {epoch_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Testing on future time points
        all_predictions = {}
        
        # Continue with the testing logic...
        # (remainder of the function for prediction follows a similar pattern)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': all_predictions, 
            'train_time_points': train_time_points,
            'test_time_points': test_time_points
        }
    
    def _validate(
        self,
        val_cells,
        gene_expression_data,
        gene_expression_history,
        cell_positions,
        time_points,
        distance_threshold
    ):
        """
        Validate the model on validation cells
        
        Args:
            val_cells: List of validation cell IDs
            gene_expression_data: Dictionary of gene expression data
            gene_expression_history: Dictionary of gene expression history
            cell_positions: Dictionary of cell positions
            time_points: List of time points to validate on
            distance_threshold: Maximum distance to consider cells as neighbors
            
        Returns:
            val_loss: Average validation loss
        """
        self.model.eval()
        val_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for t in time_points:
                cell_graphs = {}
                
                for cell_id in val_cells:
                    # Create base graph
                    base_graph = self.graph_constructor.construct_base_graph(cell_id)
                    
                    # Update with neighbor information
                    updated_graph = self.graph_constructor.update_graph_with_neighbors(
                        base_graph, cell_id, cell_positions, t,
                        distance_threshold=distance_threshold
                    )
                    
                    # Assign node features with time lags
                    graph_data = self.graph_constructor.assign_node_features(
                        updated_graph, cell_id, t, gene_expression_history,
                        self.delta_gl, self.delta_lr, self.delta_rg, self.delta_gg
                    )
                    
                    cell_graphs[cell_id] = graph_data.to(self.device)
                
                # Forward pass
                batch_graphs = [cell_graphs[cell_id] for cell_id in val_cells]
                predictions, _ = self.model(batch_graphs, gene_expression_history, cell_positions)
                
                # Calculate loss
                for i, cell_id in enumerate(val_cells):
                    for gene_idx in range(self.num_genes):
                        if gene_idx in predictions[i]:
                            pred = predictions[i][gene_idx]
                            target = torch.tensor(
                                [[gene_expression_data[cell_id][gene_idx][t]]],
                                dtype=torch.float, device=self.device
                            )
                            val_loss += self.criterion(pred, target).item()
                            num_samples += 1
        
        return val_loss / num_samples if num_samples > 0 else float('inf')
    
    def _predict(
        self,
        cell_ids,
        gene_expression_data,
        gene_expression_history,
        cell_positions,
        time_points,
        distance_threshold
    ):
        """
        Make predictions for all cells
        
        Args:
            cell_ids: List of cell IDs
            gene_expression_data: Dictionary of gene expression data
            gene_expression_history: Dictionary of gene expression history
            cell_positions: Dictionary of cell positions
            time_points: List of time points to predict
            distance_threshold: Maximum distance to consider cells as neighbors
            
        Returns:
            all_predictions: Dictionary of predicted gene expression values
        """
        self.model.eval()
        all_predictions = {cell_id: {gene_idx: {} for gene_idx in range(self.num_genes)} 
                          for cell_id in cell_ids}
        
        with torch.no_grad():
            for t in time_points:
                cell_graphs = {}
                
                for cell_id in cell_ids:
                    # Create base graph
                    base_graph = self.graph_constructor.construct_base_graph(cell_id)
                    
                    # Update with neighbor information
                    updated_graph = self.graph_constructor.update_graph_with_neighbors(
                        base_graph, cell_id, cell_positions, t,
                        distance_threshold=distance_threshold
                    )
                    
                    # Assign node features with time lags
                    graph_data = self.graph_constructor.assign_node_features(
                        updated_graph, cell_id, t, gene_expression_history,
                        self.delta_gl, self.delta_lr, self.delta_rg, self.delta_gg
                    )
                    
                    cell_graphs[cell_id] = graph_data.to(self.device)
                
                # Forward pass
                batch_graphs = [cell_graphs[cell_id] for cell_id in cell_ids]
                predictions, attention_weights = self.model(batch_graphs, gene_expression_history, cell_positions)
                
                # Store predictions
                for i, cell_id in enumerate(cell_ids):
                    for gene_idx in range(self.num_genes):
                        if gene_idx in predictions[i]:
                            all_predictions[cell_id][gene_idx][t] = predictions[i][gene_idx].item()
                
                # Update gene expression history with predictions
                for cell_id in cell_ids:
                    cell_idx = cell_ids.index(cell_id)
                    if cell_idx in predictions:
                        for gene_idx, pred in predictions[cell_idx].items():
                            gene_expression_history[cell_id][gene_idx][t] = pred.item()
        
        return all_predictions 