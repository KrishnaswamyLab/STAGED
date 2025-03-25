from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig, OmegaConf

import torch

@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # Define paths
    features_path = os.path.join(cfg.data.processed_data_dir,"features.csv")
    labels_path = os.path.join(cfg.data.processed_data_dir, "labels.csv")
    model_path = os.path.join(cfg.model.models_dir, "model.pkl")


    # Training logic
    logger.info(f"Training model with features: {features_path}, labels: {labels_path}, saving to {model_path}...")

    ######### Our objective is to have as input the GRN and the spatial graph at time step 0 and predict for t+1
    ######### We then want to uncover cell-cell interactions using the attention from the learned weights.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ###### DATA RETRIEVAL ######

    dataset = SpatialTemporalDataset(features_path, labels_path)

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    ###### Retrieve the spatial data ######
    ## Create a spatial graph from spatial data.

    #### RETRIEVE THE time-series GRN for every cell (or cell type)####
    ## We might need MIOFlow to retrieve the gene evolution for each cell type.
    ### We can use RITINI to retrieve those.


    ###### STAGED module (we want to predict the next GRN state of the cell-cell graph)
    ### Input the original GRAPH-time series into the model
    ### Loss function it trying to predict next time point for the nighbors.
    ### Retrieve edges between nodes.

    logger.success("Model training complete.")
    # return metric_value
    return


if __name__ == "__main__":
    main()

