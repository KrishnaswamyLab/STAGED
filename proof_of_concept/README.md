# STAGED: Spatiotemporal Analysis of Gene Expression Dynamics

This is a proof-of-concept implementation of the STAGED algorithm for predicting gene expression trajectories using spatial information and gene regulatory networks (GRNs).

## Overview

STAGED implements a graph-based approach to model gene expression dynamics in spatial contexts, taking into account:

1. Cell-type-specific gene regulatory networks (GRNs)
2. Ligand-receptor interactions between cells
3. Spatial proximity of cells
4. Temporal dynamics with appropriate time lags

The model uses Graph Attention Networks (GAT) to learn the influence of genes on each other and predict future gene expression values.

## Project Structure

```
proof_of_concept/
├── models/
│   └── staged.py         # Main STAGED model implementation
├── utils/
│   ├── data_utils.py     # Utilities for data loading and preprocessing
│   ├── graph_constructor.py  # Utilities for constructing cell-specific graphs
│   └── visualization.py  # Utilities for visualizing results
├── main.py               # Main script to run the model
├── trainer.py            # Training procedures
└── README.md             # This file
```

## Requirements

- Python 3.7+
- PyTorch 1.8+
- PyTorch Geometric
- NetworkX
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Usage

To run the model with default parameters and synthetic data:

```bash
python main.py --visualize
```

### Command-line Arguments

```
--expression_data: Path to gene expression data file
--positions_data: Path to cell position data file
--lr_pairs_data: Path to ligand-receptor pairs data file
--cell_types_data: Path to cell type assignments data file
--prior_grns_data: Path to prior GRNs data file
--hidden_dim: Hidden dimension for the model (default: 64)
--num_gat_layers: Number of GAT layers (default: 2)
--num_mlp_layers: Number of MLP layers (default: 2)
--delta_gl: Time lag for gene -> ligand (default: 1)
--delta_lr: Time lag for ligand -> receptor (default: 1)
--delta_rg: Time lag for receptor -> gene (default: 1)
--delta_gg: Time lag for gene -> gene (default: 1)
--num_epochs: Number of epochs to train for (default: 50)
--batch_size: Batch size (default: 16)
--learning_rate: Learning rate (default: 0.001)
--weight_decay: Weight decay (default: 1e-5)
--patience: Patience for early stopping (default: 10)
--validation_split: Fraction of data to use for validation (default: 0.1)
--distance_threshold: Maximum distance to consider cells as neighbors (default: 10.0)
--visualize: Visualize results
--output_dir: Output directory for results and visualizations (default: 'results')
--device: Device to run the model on (default: 'cuda' if available, else 'cpu')
```

## Model Components

### STAGED Model (models/staged.py)

The core model that implements the Graph Attention Network (GAT) to learn cell-specific gene interactions and predict gene expression.

### Graph Constructor (utils/graph_constructor.py)

Utilities for constructing cell-specific graphs that incorporate:
- Cell-type-specific GRNs
- Receptor nodes for each receptor gene
- Ligand nodes for ligand genes
- Connections to neighboring cells based on spatial proximity

### Trainer (trainer.py)

The main training procedure that handles:
- Data preprocessing
- Graph construction for each cell and time point
- Model training with appropriate time lags
- Validation and prediction

### Visualization Utilities (utils/visualization.py)

Functions for visualizing:
- Gene expression trajectories
- Spatial distribution of gene expression
- Graph structures with attention weights
- Training loss curves
- Gene correlations

## Data Format

The model expects data in the following format:

1. Gene expression data: Dictionary mapping cell IDs to gene expression trajectories
2. Cell positions: Dictionary mapping cell IDs to spatial positions at each time point
3. Ligand-receptor pairs: List of (ligand, receptor) gene pairs
4. Cell type assignments: Dictionary mapping cell IDs to cell types
5. Prior GRNs: Dictionary mapping cell types to prior GRNs

For this proof-of-concept, synthetic data is generated if no input files are provided.

## Citation

If you use this code in your research, please cite:

```
@article{STAGED2023,
  title={STAGED: Spatiotemporal Analysis of Gene Expression Dynamics},
  author={Your Name},
  journal={Journal Name},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 