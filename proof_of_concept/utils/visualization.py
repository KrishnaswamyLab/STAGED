import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import seaborn as sns

def visualize_attention_graph(pyg_graph, edge_index, attention_weights, pos):
    """
    Visualize a graph with attention weights and node types.
    
    Args:
        pyg_graph: PyTorch Geometric graph object containing node names and types
        edge_index: Tensor containing edge indices
        attention_weights: Tensor containing attention weights
            from `node_embeddings, (edge_index, attention_weights) = model(pyg_graph)`
        pos: Dictionary mapping node names to positions
        
    Returns:
        None. Displays the graph visualization.
    """
    # Create a NetworkX graph from edge_index with node names
    G = nx.DiGraph()
    node_names = pyg_graph.node_names
    edge_list = [(node_names[src], node_names[dst]) for src, dst in edge_index.t().tolist()]
    G.add_edges_from(edge_list)

    # Compute average attention weights across heads for visualization
    avg_attention = attention_weights.mean(dim=1).detach().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create color map for node types
    node_type_colors = {
        'gene': 'lightblue',
        'receptor': 'lightgreen', 
        'ligand': 'orange',
        'input_ligand': 'salmon'
    }

    # Get node colors based on type
    node_colors = [node_type_colors[node_type] for node_type in pyg_graph.node_types]

    # Draw graph with attention weights as edge colors and labels
    nx.draw(G, pos, ax=ax, with_labels=True, node_color=node_colors,
            edge_color=avg_attention, edge_cmap=plt.cm.Blues, width=2,
            labels={node: node for node in G.nodes()})

    # Add edge labels showing attention weights
    # Create edge labels dictionary including self-loops
    edge_labels = {}
    for (src, dst), att in zip(edge_index.t().tolist(), avg_attention):
        src_name = node_names[src]
        dst_name = node_names[dst]
        edge_labels[(src_name, dst_name)] = f'{att:.3f}'
        # For self-loops, adjust position slightly to make label visible
        if src == dst:
            pos_adj = {node: (x + 0.1, y + 0.1) for node, (x, y) in pos.items()}
            nx.draw_networkx_edge_labels(G, pos_adj, edge_labels={(src_name, dst_name): edge_labels[(src_name, dst_name)]}, ax=ax)
    
    # Draw labels for non-self-loop edges
    non_self_loops = {(s, d): l for (s, d), l in edge_labels.items() if s != d}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=non_self_loops, ax=ax)

    # Add colorbar for attention weights
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                              norm=plt.Normalize(vmin=avg_attention.min(),
                                               vmax=avg_attention.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Average Attention Weight')

    # Add legend for node types
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, label=node_type, markersize=10)
                      for node_type, color in node_type_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right', title='Node Types')

    ax2 = ax.twinx()
    ax2.set_yticks([])

    ax.set_title('Graph with Attention Weights and Values')

    plt.tight_layout()
    plt.show()

def plot_gene_trajectories(gene_expression_data, predictions, cell_id, gene_indices, figsize=(12, 8)):
    """
    Plot the actual and predicted gene expression trajectories for a cell
    
    Args:
        gene_expression_data: Dictionary of actual gene expression data
        predictions: Dictionary of predicted gene expression values
        cell_id: Cell ID to plot
        gene_indices: List of gene indices to plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    time_points = sorted(gene_expression_data[cell_id][gene_indices[0]].keys())
    
    for i, gene_idx in enumerate(gene_indices):
        plt.subplot(len(gene_indices), 1, i+1)
        
        # Actual trajectory
        actual = [gene_expression_data[cell_id][gene_idx][t] for t in time_points]
        plt.plot(time_points, actual, 'b-', label='Actual')
        
        # Predicted trajectory if available
        if cell_id in predictions and gene_idx in predictions[cell_id]:
            pred_time_points = sorted(predictions[cell_id][gene_idx].keys())
            pred = [predictions[cell_id][gene_idx][t] for t in pred_time_points]
            plt.plot(pred_time_points, pred, 'r--', label='Predicted')
        
        plt.title(f"Gene {gene_idx}")
        plt.xlabel("Time")
        plt.ylabel("Expression")
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"gene_trajectories_{cell_id}.png")
    plt.close()


def plot_spatial_expression(cell_positions, gene_expression_data, time_point, gene_idx, 
                          figsize=(10, 8), cmap='viridis'):
    """
    Plot the spatial distribution of gene expression at a specific time point
    
    Args:
        cell_positions: Dictionary of cell positions
        gene_expression_data: Dictionary of gene expression data
        time_point: Time point to visualize
        gene_idx: Gene index to visualize
        figsize: Figure size
        cmap: Colormap for expression values
    """
    plt.figure(figsize=figsize)
    
    # Extract cell positions at the specified time point
    x_coords = []
    y_coords = []
    expressions = []
    
    for cell_id, positions in cell_positions.items():
        if time_point in positions and cell_id in gene_expression_data:
            if gene_idx in gene_expression_data[cell_id]:
                if time_point in gene_expression_data[cell_id][gene_idx]:
                    x, y = positions[time_point]
                    expression = gene_expression_data[cell_id][gene_idx][time_point]
                    
                    x_coords.append(x)
                    y_coords.append(y)
                    expressions.append(expression)
    
    # Create a scatter plot with expression values as colors
    scatter = plt.scatter(x_coords, y_coords, c=expressions, cmap=cmap, 
                     s=100, alpha=0.8, edgecolors='k')
    
    plt.colorbar(scatter, label="Expression level")
    plt.title(f"Spatial distribution of Gene {gene_idx} at time {time_point}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(f"spatial_expression_t{time_point}_g{gene_idx}.png")
    plt.close()


def animate_gene_expression(cell_positions, gene_expression_data, gene_idx, 
                         output_file='gene_expression_animation.mp4', figsize=(10, 8)):
    """
    Create an animation of gene expression over time
    
    Args:
        cell_positions: Dictionary of cell positions
        gene_expression_data: Dictionary of gene expression data
        gene_idx: Gene index to visualize
        output_file: Output file path
        figsize: Figure size
    """
    # Determine time points
    all_time_points = set()
    for cell_id, positions in cell_positions.items():
        all_time_points.update(positions.keys())
    time_points = sorted(all_time_points)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine expression range for consistent colormap
    all_expressions = []
    for cell_id, genes_data in gene_expression_data.items():
        if gene_idx in genes_data:
            all_expressions.extend(genes_data[gene_idx].values())
    vmin = min(all_expressions)
    vmax = max(all_expressions)
    
    # Animation update function
    def update(frame):
        ax.clear()
        time_point = time_points[frame]
        
        # Extract cell positions and expressions for this time point
        x_coords = []
        y_coords = []
        expressions = []
        
        for cell_id, positions in cell_positions.items():
            if time_point in positions and cell_id in gene_expression_data:
                if gene_idx in gene_expression_data[cell_id]:
                    if time_point in gene_expression_data[cell_id][gene_idx]:
                        x, y = positions[time_point]
                        expression = gene_expression_data[cell_id][gene_idx][time_point]
                        
                        x_coords.append(x)
                        y_coords.append(y)
                        expressions.append(expression)
        
        # Create scatter plot
        scatter = ax.scatter(x_coords, y_coords, c=expressions, cmap='viridis', 
                         s=100, alpha=0.8, edgecolors='k', vmin=vmin, vmax=vmax)
        
        ax.set_title(f"Gene {gene_idx} expression at time {time_point}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.grid(True, linestyle='--', alpha=0.6)
        
        return scatter,
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(time_points), blit=True)
    
    # Save the animation
    anim.save(output_file, writer='ffmpeg', fps=2)
    plt.close()


def plot_attention_weights(graph, attention_weights, cell_id, time_point, figsize=(12, 10)):
    """
    Visualize attention weights between nodes in the graph
    
    Args:
        graph: NetworkX graph
        attention_weights: Attention weights from the GAT layer
        cell_id: Cell ID
        time_point: Time point
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Create a layout for the graph
    pos = nx.spring_layout(graph)
    
    # Extract edge indices and weights
    if isinstance(attention_weights, tuple):
        edge_index, weights = attention_weights
        edge_index = edge_index.cpu().numpy()
        weights = weights.cpu().numpy()
    else:
        # If we don't have attention weights, use equal weights for all edges
        edges = list(graph.edges())
        edge_index = np.array([[i, j] for i, j in edges]).T
        weights = np.ones(len(edges))
    
    # Normalize weights for visualization
    if len(weights) > 0:
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    
    # Draw the nodes
    nx.draw_networkx_nodes(graph, pos, node_size=700, node_color='lightblue', alpha=0.8)
    
    # Draw the edges with attention weights determining width
    for i in range(edge_index.shape[1]):
        source, target = edge_index[0, i], edge_index[1, i]
        weight = weights[i] if i < len(weights) else 0.1
        nx.draw_networkx_edges(
            graph, pos, 
            edgelist=[(source, target)], 
            width=weight * 5,  # Scale weight for visibility
            alpha=weight,
            arrows=True, 
            arrowstyle='-|>', 
            arrowsize=20
        )
    
    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight='bold')
    
    plt.title(f"Graph structure with attention weights for cell {cell_id} at time {time_point}")
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(f"attention_graph_cell{cell_id}_t{time_point}.png")
    plt.close()


def plot_training_curves(losses, figsize=(10, 6)):
    """
    Plot training and validation loss curves
    
    Args:
        losses: Dictionary with 'train_losses' and 'val_losses'
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(losses['train_losses']) + 1)
    
    plt.plot(epochs, losses['train_losses'], 'b-', label='Training loss')
    plt.plot(epochs, losses['val_losses'], 'r-', label='Validation loss')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig("training_curves.png")
    plt.close()


def plot_gene_correlations(gene_expression_data, cell_ids, gene_indices, time_point,
                         figsize=(12, 10)):
    """
    Plot correlations between genes at a specific time point
    
    Args:
        gene_expression_data: Dictionary of gene expression data
        cell_ids: List of cell IDs to include
        gene_indices: List of gene indices to include
        time_point: Time point to analyze
        figsize: Figure size
    """
    # Extract gene expression values for the specified genes and time point
    data = []
    for gene_idx in gene_indices:
        gene_data = []
        for cell_id in cell_ids:
            if cell_id in gene_expression_data and gene_idx in gene_expression_data[cell_id]:
                if time_point in gene_expression_data[cell_id][gene_idx]:
                    gene_data.append(gene_expression_data[cell_id][gene_idx][time_point])
        data.append(gene_data)
    
    # Convert to numpy array
    data = np.array(data)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data)
    
    # Plot correlation heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
              xticklabels=[f"Gene {g}" for g in gene_indices],
              yticklabels=[f"Gene {g}" for g in gene_indices])
    
    plt.title(f"Gene Expression Correlations at Time {time_point}")
    plt.tight_layout()
    
    plt.savefig(f"gene_correlations_t{time_point}.png")
    plt.close() 