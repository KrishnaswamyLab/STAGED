import os
import pickle
import torch
import numpy as np
import anndata
import scanpy as sc
import pandas as pd

def retrieve_simulated_data(data_dir="data/raw",sim_file="simulation_results.pkl"):
    """
    Load simulated data from the specified directory.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing simulated data files
        
    Returns:
    --------
    dict
        Dictionary containing all simulated data components
    """
    # Create an empty dictionary to store loaded data
    data = {}
    
    # Verify the directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load the simulation results
    with open(os.path.join(data_dir,sim_file), 'rb') as f:
        sim_data = pickle.load(f)
    # Extract data from the loaded simulation results
    # Based on the saving function structure:
    # - 'genes' is a 3D array (time_points x cells x genes)
    # - 'positions' is a 3D array (time_points x cells x 2)
    # - 'metadata' contains time_points, cell_ids, gene_names, cell_types, and prior_grns
    
    # Extract gene expression data (time_points x cells x genes)
    data['gene_expression'] = torch.tensor(sim_data['genes'],dtype=torch.float32)
    # Extract cell positions (time_points x cells x 2)
    data['cell_positions'] = torch.tensor(sim_data['positions'],dtype=torch.float32)

    # Extract metadata
    metadata = sim_data['metadata']
    
    # Extract gene names
    data['genes'] = metadata['gene_names']

    # Extract cell type assignments
    cell_ids = metadata['cell_ids']
    cell_types_dict = metadata['cell_types']
    
    # Create a mapping from cell IDs to their corresponding types
    unique_cell_types = sorted(set(cell_types_dict.values()))
    label_to_int = {label: idx for idx, label in enumerate(unique_cell_types)}
    # Map each cell ID to its corresponding integer label
    assignments = [label_to_int[cell_types_dict[cell_id]] for cell_id in cell_ids]

    data['cell_type_assignments'] = torch.tensor(assignments, dtype=torch.long)
    
    # Extract prior GRNs
    cell_specific_prior_grns =  [metadata['prior_grns'][cell_type] for cell_type in label_to_int.keys()]
    data['prior_grns'] = cell_specific_prior_grns

    data['receptor_gene_pairs'] = metadata['receptor_gene_pairs']
    data['ligand_receptor_pairs'] = metadata['ligand_receptor_pairs']

    valid_genes = set(data['genes'])
    print(f"Valid genes in simulation: {sorted(valid_genes)}")

    original_lr_pairs = data['ligand_receptor_pairs']
    filtered_lr_pairs = []
    for ligand, receptor in original_lr_pairs:
        if ligand in valid_genes and receptor in valid_genes:
            filtered_lr_pairs.append((ligand, receptor))
        else:
            print(f"Removing invalid L-R pair: ({ligand}, {receptor})")

    data['ligand_receptor_pairs'] = filtered_lr_pairs

    # Filter receptor_gene_pairs to only include existing genes  
    original_rg_pairs = data['receptor_gene_pairs']
    filtered_rg_pairs = []
    for receptor, gene in original_rg_pairs:
        if receptor in valid_genes and gene in valid_genes:
            filtered_rg_pairs.append((receptor, gene))
        else:
            print(f"Removing invalid R-G pair: ({receptor}, {gene})")

    data['receptor_gene_pairs'] = filtered_rg_pairs


     # Calculate dimensions
    data['n_time_points'] = data['gene_expression'].shape[0]
    data['n_cells'] = data['gene_expression'].shape[1]
    data['n_genes'] = data['gene_expression'].shape[2]


    return data

def retrieve_real_data(data_dir="data/real"):
    # Create an empty dictionary to store loaded data
    data = {}
    
    # Verify the directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    traj_data = np.load(os.path.join(data_dir,'trajectories.npz'), allow_pickle=True)
    gene_expression =  np.load(os.path.join(data_dir,'filtered_trajectories.npz'), allow_pickle=True)

    # Extract gene expression data (time_points x cells x genes)
    data['gene_expression'] = torch.tensor(gene_expression,dtype=torch.float32)

    # Extract cell positions (time_points x cells x 2)
    # data['cell_positions'] = torch.tensor(traj_data['positions'],dtype=torch.float32)
    # data['cell_positions'] = torch.tensor(traj_data['positions'],dtype=torch.float32)
    # data['cell_positions'] = torch.zeros(traj_data['trajectories'].shape[0],traj_data['trajectories'].shape[1],2)

    # Generate random cell positions in 10x10 space instead of zeros
    n_time_points = traj_data['trajectories'].shape[0]
    n_cells = traj_data['trajectories'].shape[1]
    base_positions = torch.rand(n_cells, 2) * 10
    random_positions = base_positions.unsqueeze(0).expand(n_time_points, -1, -1)
    # data['cell_positions'] = torch.tensor(sim_data['positions'],dtype=torch.float32)
    data['cell_positions'] = random_positions

    # Extract gene names
    gene_list = np.load(os.path.join(data_dir,'filtered_genes.npy'), allow_pickle=True)
    data['genes'] = gene_list.tolist()

    data['cell_type_assignments'] = torch.tensor(traj_data['annotations'], dtype=torch.long)

    # Extract prior GRNs
    with open(os.path.join(data_dir,'prior_graphs.pkl'), 'rb') as f:
        loaded_graphs = pickle.load(f)
    data['prior_grns'] = loaded_graphs

    data['receptor_gene_pairs'] = [('PLXNA4', 'PLXNA4'), ('ADGRL2', 'ADGRL2'),('MPZL1', 'MPZL1')]
    data['ligand_receptor_pairs'] = [('APP', 'PLXNA4'), ('TENM4', 'ADGRL2'), ('MPZL1', 'MPZL1')]

     # Calculate dimensions
    data['n_time_points'] = data['gene_expression'].shape[0]
    data['n_cells'] = data['gene_expression'].shape[1]
    data['n_genes'] = data['gene_expression'].shape[2]

    #Subsample time
    original_n_timepoints = data['n_time_points']
    n_selected = 10
    selected_indices = np.linspace(0, original_n_timepoints - 1, n_selected, dtype=int).tolist()


    subsampled_data = data.copy()
    # Apply subsampling to all time-dependent data
    subsampled_data['gene_expression'] = data['gene_expression'][selected_indices]
    subsampled_data['cell_positions'] = data['cell_positions'][selected_indices]
    
    # Update dimension
    subsampled_data['n_time_points'] = n_selected
    
    return subsampled_data

if __name__ == "__main__":
    # Retrieve simulated data
    sim_data = retrieve_simulated_data()
    print(sim_data['cell_type_assignments'] )
    print(sim_data['prior_grns'])
    print(sim_data['receptor_gene_pairs'])
    print(sim_data['ligand_receptor_pairs'])
    print(sim_data['cell_type_assignments'])
    data = retrieve_real_data()
    # print(data)
    print(data['prior_grns'])
    print(data['receptor_gene_pairs'])
    print(data['ligand_receptor_pairs'])
    print(data['cell_type_assignments'])
    print(data['cell_positions'])


    # Ensure the processed data directory exists
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Save each component of the data dictionary as a separate pickle file
    for key, value in data.items():
        file_path = os.path.join(processed_dir, f"{key}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(value, f)

def retrieve_axalotl_data(data_dir="data/axalotl", processed_file="axalotl_processed.pkl"):
    """
    Load precomputed axolotl data. Run notebooks/precompute_axolotl.py first
    to generate the processed pickle.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    path = os.path.join(data_dir, processed_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed axolotl file not found: {path}. "
            f"Run notebooks/precompute_axolotl.py to generate it."
        )

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    data = {
        'gene_expression':       torch.tensor(bundle['gene_expression'], dtype=torch.float32),
        'cell_positions':        torch.tensor(bundle['cell_positions'], dtype=torch.float32),
        'genes':                 bundle['genes'],
        'cell_type_assignments': torch.tensor(bundle['cell_type_assignments'], dtype=torch.long),
        'prior_grns':            bundle['prior_grns'],   # dict[str, nx.DiGraph], pass through
        'ligand_receptor_pairs': bundle['ligand_receptor_pairs'],
        'receptor_gene_pairs':   bundle['receptor_gene_pairs'],
    }
    data['n_time_points'] = data['gene_expression'].shape[0]
    data['n_cells']       = data['gene_expression'].shape[1]
    data['n_genes']       = data['gene_expression'].shape[2]
    return data

# def retrieve_axalotl_data(data_dir="data/axalotl"):
#     """
#     Load real axolotl spatial-transcriptomics data.

#     Expects `data_dir` to contain:
#         - axolotl_ran.h5ad   AnnData with .obs['Annotation']
#         - strajs.npy         spatial trajectories  (time_points, cells, 2)
#         - gtrajs.npy         gene-expr trajectories (time_points, cells, all_genes)
#     """


#     if not os.path.exists(data_dir):
#         raise FileNotFoundError(f"Data directory not found: {data_dir}")

#     data = {}
#     # 1. Load AnnData + precomputed trajectories
#     adata = anndata.read_h5ad(os.path.join(data_dir, "axolotl_ran.h5ad"))
#     spatial_traj = np.load(os.path.join(data_dir, "strajs.npy"), allow_pickle=True)
#     gene_traj    = np.load(os.path.join(data_dir, "gtrajs.npy"), allow_pickle=True)

#     print("I AM HERE 1")

#     # 2. Highly variable genes (top 100, seurat flavor)
#     sc.pp.highly_variable_genes(
#         adata, flavor='seurat', n_top_genes=100, subset=False, inplace=True,
#     )
#     hvg_mask  = adata.var['highly_variable'].values
#     hvg_genes = adata.var_names[hvg_mask]
#     n_hvg     = len(hvg_genes)

#     print("I AM HERE 2")

    
#     # 3. Gene expression: tensor + subset to HVGs
#     gene_traj_hvg = gene_traj[:, :, hvg_mask]   
#     data['gene_expression'] = torch.from_numpy(
#         np.ascontiguousarray(gene_traj_hvg, dtype=np.float32)
#     )

#     print("I AM HERE 3")


#     # 4. Spatial positions
#     data['cell_positions'] = torch.tensor(spatial_traj, dtype=torch.float32)

#     print("I AM HERE 4")

#     # 5. Gene names
#     data['genes'] = hvg_genes.tolist()

#     # 6. Cell-type assignments
#     cell_types        = adata.obs['Annotation'].values
#     unique_cell_types = sorted(set(cell_types))
#     label_to_int      = {label: idx for idx, label in enumerate(unique_cell_types)}
#     assignments       = [label_to_int[label] for label in cell_types]
#     data['cell_type_assignments'] = torch.tensor(assignments, dtype=torch.long)

#     print("I AM HERE 6")

#     # 7. Prior GRNs: fully connected, no self-loops, one per cell type
#     full_grn = np.ones((n_hvg, n_hvg), dtype=np.float32)
#     np.fill_diagonal(full_grn, 0.0)
#     full_grn_tensor = torch.tensor(full_grn, dtype=torch.float32)
#     data['prior_grns'] = [full_grn_tensor.clone() for _ in unique_cell_types]

#     print("I AM HERE 7")


#     # 8. L-R pairs from the ARTISTA CSV; R-G pairs are each receptor paired to itself
#     lr_df = pd.read_csv("/nfs/roberts/project/pi_sk2433/bp542/Axolotl_Spatial/ARTISTA_LR_pairs.csv")
#     ligand_receptor_pairs = list(zip(lr_df['Ligand'], lr_df['Receptor']))
#     receptor_gene_pairs   = [(r, r) for r in sorted(set(lr_df['Receptor']))]

#     valid_genes = set(data['genes'])
#     print(f"Valid genes in axolotl data: {len(valid_genes)} HVGs")
#     print(f"Loaded {len(ligand_receptor_pairs)} L-R pairs from ARTISTA CSV")

#     filtered_lr_pairs = []
#     dropped_lr = 0
#     for ligand, receptor in ligand_receptor_pairs:
#         if ligand in valid_genes and receptor in valid_genes:
#             filtered_lr_pairs.append((ligand, receptor))
#         else:
#             dropped_lr += 1
#     data['ligand_receptor_pairs'] = filtered_lr_pairs

#     filtered_rg_pairs = []
#     dropped_rg = 0
#     for receptor, gene in receptor_gene_pairs:
#         if receptor in valid_genes and gene in valid_genes:
#             filtered_rg_pairs.append((receptor, gene))
#         else:
#             dropped_rg += 1
#     data['receptor_gene_pairs'] = filtered_rg_pairs

#     print(f"L-R pairs after HVG filter: {len(filtered_lr_pairs)} kept, {dropped_lr} dropped")
#     print(f"R-G pairs after HVG filter: {len(filtered_rg_pairs)} kept, {dropped_rg} dropped")

#     # 9. Dimensions
#     data['n_time_points'] = data['gene_expression'].shape[0]
#     data['n_cells']       = data['gene_expression'].shape[1]
#     data['n_genes']       = data['gene_expression'].shape[2]

#     return data