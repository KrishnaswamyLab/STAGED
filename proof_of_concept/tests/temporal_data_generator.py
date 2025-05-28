import torch
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List


def create_oscillatory_dynamics_data(
    n_time_points: int = 20,
    n_cells: int = 7,
    n_genes: int = 6,
    dt: float = 0.5,
    noise_level: float = 0.1,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """
    Create synthetic data with realistic temporal dynamics for ODE testing.
    
    Features:
    - Oscillatory gene expression patterns with different frequencies
    - Gene regulatory interactions that affect dynamics
    - Spatial cell communication effects
    - Realistic noise levels
    
    Args:
        n_time_points: Number of time points
        n_cells: Number of cells (default 7 for hex grid)
        n_genes: Number of genes
        dt: Time step size
        noise_level: Gaussian noise standard deviation
        device: Device to create tensors on
        
    Returns:
        Dictionary with gene_expression, cell_positions, and metadata
    """
    
    # Time points
    time_points = torch.arange(n_time_points, dtype=torch.float32) * dt
    
    # Initialize gene expression tensor
    gene_expression = torch.zeros((n_time_points, n_cells, n_genes), device=device)
    
    # Define oscillatory patterns for each gene
    base_frequencies = torch.tensor([1.0, 1.5, 0.8, 2.0, 0.5, 1.2])[:n_genes]
    base_amplitudes = torch.tensor([1.0, 0.8, 1.2, 0.6, 1.5, 0.9])[:n_genes]
    phase_shifts = torch.tensor([0.0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, np.pi/3])[:n_genes]
    
    # Create hexagonal cell positions (static for simplicity)
    cell_positions = torch.zeros((n_time_points, n_cells, 2), device=device)
    hex_distance = 10.0
    angle_step = 2 * np.pi / 6
    
    for t in range(n_time_points):
        # Center cell
        cell_positions[t, 0] = torch.tensor([0.0, 0.0])
        # Surrounding cells in hex pattern
        for i in range(6):
            if i + 1 < n_cells:
                angle = i * angle_step
                x = hex_distance * np.cos(angle)
                y = hex_distance * np.sin(angle)
                cell_positions[t, i+1] = torch.tensor([x, y])
    
    # Generate base dynamics for each cell and gene
    for t_idx, t in enumerate(time_points):
        for cell_idx in range(n_cells):
            for gene_idx in range(n_genes):
                # Base oscillatory pattern
                base_value = base_amplitudes[gene_idx] * torch.sin(
                    base_frequencies[gene_idx] * t + phase_shifts[gene_idx]
                )
                
                # Cell-specific modulation (center cell has different dynamics)
                if cell_idx == 0:  # Center cell
                    cell_modulation = 1.0
                else:  # Surrounding cells
                    # Add slight variation based on position
                    angle = (cell_idx - 1) * angle_step
                    cell_modulation = 0.8 + 0.4 * np.cos(angle + t * 0.5)
                
                # Gene regulatory effects (simple coupling)
                if gene_idx > 0:
                    # Current gene is influenced by previous gene with delay
                    if t_idx > 2:  # Ensure we have history
                        regulatory_effect = 0.2 * gene_expression[t_idx-2, cell_idx, gene_idx-1]
                    else:
                        regulatory_effect = 0.0
                else:
                    regulatory_effect = 0.0
                
                # Spatial communication effect (ligand-receptor like)
                spatial_effect = 0.0
                if cell_idx > 0 and gene_idx in [1, 3]:  # Receptor genes
                    # Get signal from center cell
                    if t_idx > 1:
                        ligand_gene = 0 if gene_idx == 1 else 2
                        spatial_effect = 0.15 * gene_expression[t_idx-1, 0, ligand_gene]
                
                # Combine all effects
                final_value = (
                    base_value * cell_modulation +
                    regulatory_effect +
                    spatial_effect +
                    1.0  # Baseline offset
                )
                
                # Add noise
                noise = torch.normal(0, noise_level, (1,)).item()
                final_value += noise
                
                # Ensure non-negative (like real gene expression)
                gene_expression[t_idx, cell_idx, gene_idx] = torch.clamp(final_value, min=0.0)
    
    # Create metadata
    genes = [f"gene_{i}" for i in range(n_genes)]
    
    # Cell type assignments (alternating pattern)
    cell_type_assignments = torch.zeros(n_cells, dtype=torch.long, device=device)
    for i in range(n_cells):
        cell_type_assignments[i] = i % 2
    
    # Ligand-receptor pairs
    ligand_receptor_pairs = [
        ("gene_0", "gene_1"),  # gene_0 (ligand) -> gene_1 (receptor)
        ("gene_2", "gene_3"),  # gene_2 (ligand) -> gene_3 (receptor)
    ]
    
    # Receptor-gene pairs (regulatory connections)
    receptor_gene_pairs = [
        ("gene_1", "gene_4"),  # receptor gene_1 regulates gene_4
        ("gene_1", "gene_5"),  # receptor gene_1 regulates gene_5
        ("gene_3", "gene_2"),  # receptor gene_3 regulates gene_2
        ("gene_3", "gene_4"),  # receptor gene_3 regulates gene_4
    ]
    
    # Prior GRNs
    prior_grn_0 = nx.DiGraph()
    prior_grn_1 = nx.DiGraph()
    
    for grn in [prior_grn_0, prior_grn_1]:
        for gene in genes:
            grn.add_node(gene)
        # Add some regulatory edges
        grn.add_edge("gene_0", "gene_1")
        grn.add_edge("gene_1", "gene_2")
        grn.add_edge("gene_2", "gene_3")
        grn.add_edge("gene_3", "gene_4")
        grn.add_edge("gene_4", "gene_5")
        grn.add_edge("gene_5", "gene_0")  # Cycle
    
    prior_grns = {0: prior_grn_0, 1: prior_grn_1}
    
    return {
        'gene_expression': gene_expression,
        'cell_positions': cell_positions,
        'genes': genes,
        'cell_type_assignments': cell_type_assignments,
        'ligand_receptor_pairs': ligand_receptor_pairs,
        'receptor_gene_pairs': receptor_gene_pairs,
        'prior_grns': prior_grns,
        'n_time_points': n_time_points,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'dt': dt,
        'time_points': time_points
    }


def create_damped_oscillator_data(
    n_time_points: int = 25,
    n_cells: int = 4,
    n_genes: int = 4,
    dt: float = 0.2,
    device: torch.device = torch.device('cpu')
) -> Dict[str, torch.Tensor]:
    """
    Create data based on damped harmonic oscillators with coupling.
    This provides a good test case for Neural ODE as the underlying dynamics
    are governed by differential equations.
    
    Each gene pair (gene_i, gene_{i+1}) forms a damped harmonic oscillator:
    d²x/dt² + 2γ dx/dt + ω²x = F(other_genes)
    
    Args:
        n_time_points: Number of time points
        n_cells: Number of cells
        n_genes: Number of genes (should be even)
        dt: Time step
        device: Device for tensors
        
    Returns:
        Dictionary with synthetic data
    """
    
    time_points = torch.arange(n_time_points, dtype=torch.float32) * dt
    gene_expression = torch.zeros((n_time_points, n_cells, n_genes), device=device)
    
    # Oscillator parameters
    omega = torch.tensor([2.0, 2.0, 1.5, 1.5])[:n_genes]  # Natural frequencies
    gamma = torch.tensor([0.1, 0.1, 0.2, 0.2])[:n_genes]  # Damping coefficients
    
    # Initial conditions
    for cell_idx in range(n_cells):
        # Initial positions
        gene_expression[0, cell_idx, 0] = 1.0 + 0.2 * cell_idx
        gene_expression[0, cell_idx, 2] = 0.5 + 0.1 * cell_idx
        
        # Initial velocities (stored as differences)
        if n_time_points > 1:
            gene_expression[1, cell_idx, 1] = 0.1 * (cell_idx + 1)
            gene_expression[1, cell_idx, 3] = -0.05 * (cell_idx + 1)
    
    # Generate dynamics using finite differences (approximating the ODE)
    for t_idx in range(2, n_time_points):
        for cell_idx in range(n_cells):
            for gene_idx in range(0, n_genes, 2):  # Position genes
                if gene_idx + 1 < n_genes:
                    # Current position and velocity
                    x = gene_expression[t_idx-1, cell_idx, gene_idx]
                    v = gene_expression[t_idx-1, cell_idx, gene_idx+1]
                    
                    # Previous values for second derivative
                    x_prev = gene_expression[t_idx-2, cell_idx, gene_idx]
                    
                    # Coupling force from other oscillators
                    coupling_force = 0.0
                    for other_gene in range(0, n_genes, 2):
                        if other_gene != gene_idx:
                            coupling_force += 0.1 * gene_expression[t_idx-1, cell_idx, other_gene]
                    
                    # Spatial coupling (simple)
                    spatial_force = 0.0
                    for other_cell in range(n_cells):
                        if other_cell != cell_idx:
                            distance = abs(other_cell - cell_idx)
                            spatial_force += 0.05 * gene_expression[t_idx-1, other_cell, gene_idx] / (distance + 1)
                    
                    # Damped harmonic oscillator equation
                    # d²x/dt² = -ω²x - 2γ dx/dt + F_coupling + F_spatial
                    acceleration = (
                        -omega[gene_idx]**2 * x
                        -2 * gamma[gene_idx] * v
                        + coupling_force
                        + spatial_force
                    )
                    
                    # Update position and velocity using Verlet integration
                    new_x = 2*x - x_prev + acceleration * dt**2
                    new_v = (new_x - x_prev) / (2 * dt)
                    
                    # Store values (with some noise)
                    noise_x = torch.normal(0, 0.02, (1,)).item()
                    noise_v = torch.normal(0, 0.01, (1,)).item()
                    
                    gene_expression[t_idx, cell_idx, gene_idx] = new_x + noise_x
                    gene_expression[t_idx, cell_idx, gene_idx+1] = new_v + noise_v
    
    # Ensure non-negative values by adding offset and clipping
    gene_expression = torch.clamp(gene_expression + 2.0, min=0.0)
    
    # Create simple 2x2 grid positions
    cell_positions = torch.zeros((n_time_points, n_cells, 2), device=device)
    positions = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    for t in range(n_time_points):
        for c in range(min(n_cells, 4)):
            cell_positions[t, c] = torch.tensor(positions[c])
    
    # Metadata
    genes = [f"gene_{i}" for i in range(n_genes)]
    cell_type_assignments = torch.zeros(n_cells, dtype=torch.long)
    
    ligand_receptor_pairs = [("gene_0", "gene_1")] if n_genes >= 2 else []
    receptor_gene_pairs = [("gene_1", "gene_2")] if n_genes >= 3 else []
    
    # Simple GRN
    grn = nx.DiGraph()
    for gene in genes:
        grn.add_node(gene)
    for i in range(n_genes-1):
        grn.add_edge(f"gene_{i}", f"gene_{i+1}")
    
    prior_grns = {0: grn}
    
    return {
        'gene_expression': gene_expression,
        'cell_positions': cell_positions,
        'genes': genes,
        'cell_type_assignments': cell_type_assignments,
        'ligand_receptor_pairs': ligand_receptor_pairs,
        'receptor_gene_pairs': receptor_gene_pairs,
        'prior_grns': prior_grns,
        'n_time_points': n_time_points,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'dt': dt,
        'time_points': time_points
    } 