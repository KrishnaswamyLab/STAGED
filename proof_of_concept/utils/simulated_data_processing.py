import os
import pickle
import torch


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

     # Calculate dimensions
    data['n_time_points'] = data['gene_expression'].shape[0]
    data['n_cells'] = data['gene_expression'].shape[1]
    data['n_genes'] = data['gene_expression'].shape[2]


    return data

if __name__ == "__main__":
    # Retrieve simulated data
    data = retrieve_simulated_data()

    # Ensure the processed data directory exists
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    # Save each component of the data dictionary as a separate pickle file
    for key, value in data.items():
        file_path = os.path.join(processed_dir, f"{key}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(value, f)