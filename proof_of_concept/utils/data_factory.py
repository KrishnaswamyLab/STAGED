"""
Data Factory for STAGED Model Training

This module provides functions to create different types of synthetic data
for training and testing STAGED models.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict


from proof_of_concept.tests.temporal_data_generator import create_oscillatory_dynamics_data, create_damped_oscillator_data
from proof_of_concept.tests.test_graph_constructor import create_hex_grid_test_data, create_square_grid_data
from proof_of_concept.utils.simulated_data_processing import retrieve_simulated_data

def create_simple_sinusoidal_data(
    n_time_points: int = 20,
    n_cells: int = 6,
    n_genes: int = 8,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """Create simple sinusoidal test data for quick testing."""
    
    # Create time points
    time = torch.linspace(0, 4*np.pi, n_time_points).unsqueeze(-1).unsqueeze(-1)
    
    # Create gene expression patterns (sinusoidal with different frequencies and phases)
    gene_patterns = torch.zeros((n_time_points, n_cells, n_genes), device=device)
    for g in range(n_genes):
        freq = (g + 1) * 0.5  # Different frequencies
        phase = g * np.pi / 4  # Different phases
        amplitude = 1.0 + 0.3 * g  # Different amplitudes
        baseline = 2.0  # Positive baseline
        gene_patterns[..., g] = amplitude * torch.sin(freq * time + phase).squeeze(-1) + baseline
    
    # Ensure non-negative values
    gene_patterns = torch.clamp(gene_patterns, min=0.0)
    
    # Create cell positions in a hexagonal pattern
    cell_positions = torch.zeros((n_time_points, n_cells, 2), device=device)
    if n_cells >= 1:
        cell_positions[:, 0] = torch.tensor([0.0, 0.0])  # Center
    if n_cells >= 2:
        cell_positions[:, 1] = torch.tensor([1.0, 0.0])  # Right
    if n_cells >= 3:
        cell_positions[:, 2] = torch.tensor([-0.5, 0.866])  # Top-left
    if n_cells >= 4:
        cell_positions[:, 3] = torch.tensor([-0.5, -0.866])  # Bottom-left
    if n_cells >= 5:
        cell_positions[:, 4] = torch.tensor([0.5, 0.866])  # Top-right
    if n_cells >= 6:
        cell_positions[:, 5] = torch.tensor([0.5, -0.866])  # Bottom-right
    
    # Create metadata
    genes = [f"gene_{i}" for i in range(n_genes)]
    
    # Create ligand-receptor pairs
    ligand_receptor_pairs = []
    for i in range(0, min(n_genes//2, 3)):
        ligand_receptor_pairs.append((f"gene_{i}", f"gene_{i + n_genes//2}"))
    
    # Create receptor-gene pairs
    receptor_gene_pairs = []
    for i in range(n_genes//2, min(n_genes, n_genes//2 + 3)):
        for j in range(min(3, n_genes)):
            if j != i:
                receptor_gene_pairs.append((f"gene_{i}", f"gene_{j}"))
    
    # Cell type assignments
    cell_type_assignments = torch.zeros(n_cells, dtype=torch.long, device=device)
    for i in range(n_cells):
        cell_type_assignments[i] = i % 2
    
    # Create simple GRNs
    grn = nx.DiGraph()
    for gene in genes:
        grn.add_node(gene)
    # Add regulatory edges
    for i in range(n_genes - 1):
        grn.add_edge(f"gene_{i}", f"gene_{(i+1) % n_genes}")
    
    prior_grns = {0: grn, 1: grn.copy()}
    
    return {
        'gene_expression': gene_patterns,
        'cell_positions': cell_positions,
        'genes': genes,
        'cell_type_assignments': cell_type_assignments,
        'ligand_receptor_pairs': ligand_receptor_pairs,
        'receptor_gene_pairs': receptor_gene_pairs,
        'prior_grns': prior_grns,
        'n_time_points': n_time_points,
        'n_cells': n_cells,
        'n_genes': n_genes
    }


def get_data(data_type: str, device: torch.device) -> Dict:
    """Get training data based on specified type."""
    if data_type == "oscillatory":
        print("Creating oscillatory dynamics data...")
        return create_oscillatory_dynamics_data(
            n_time_points=25,
            n_cells=7,
            n_genes=6,
            dt=0.4,
            noise_level=0.05,
            device=device
        )
    
    elif data_type == "damped_oscillator":
        print("Creating damped oscillator data...")
        return create_damped_oscillator_data(
            n_time_points=30,
            n_cells=4,
            n_genes=4,
            dt=0.2,
            device=device
        )
    
    elif data_type == "hex_grid":
        print("Creating hex grid test data...")
        data = create_hex_grid_test_data()
        # Convert to device
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
        return data
    
    elif data_type == "square_grid":
        print("Creating square grid test data...")
        data = create_square_grid_data()
        # Convert to device
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
        return data
    
    elif data_type == "sinusoidal":
        print("Creating simple sinusoidal data...")
        return create_simple_sinusoidal_data(device=device)
    
    elif data_type== "simulated":
        print("Retriving simulated data...")
        data = retrieve_simulated_data(data_dir="data/raw",sim_file="100_simulation_results.pkl")
        # Convert to device
        for key, value in data.items():
            print(f"Converting {key} to device {device}")

            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
        return data
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def get_available_data_types():
    """Get list of available data types."""
    return ['oscillatory', 'damped_oscillator', 'hex_grid', 'square_grid', 'sinusoidal', 'simulated'] 