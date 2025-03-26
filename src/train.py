from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np


import hydra
import lightning as L
import rootutils
import torch
import pytorch_lightning as pl
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from utils.priorGraphs import retrieve_grn_priors
from utils.cellTypes import retrieve_cell_types
from src.utils.ligand_receptors import retrieve_ligands, retrieve_receptors

# PyTorch geometric
import torch_geometric
import torch_geometric.data as geom_data
import torch_geometric.nn as geom_nn

@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # Define paths
    genes_trajectories_path = os.path.join(cfg.data.processed_data_dir,"genes_trajectories.csv")
    cell_trajectories_path = os.path.join(cfg.data.processed_data_dir,"cell_trajectories.csv")

    model_path = os.path.join(cfg.model.models_dir, "model.pkl")

    ###### INPUTS ######
    # Retrieve the cells
    cell_list = retrieve_cell_list()

    # Retrieve the Prior GRNs for each cell type
    grn_priors_array = retrieve_grn_priors()

    #Retrieve the ligand genes
    receptors = retrieve_receptors()
    ligands = retrieve_ligands()


    ###### Initialize cell-type specific GRN ######
    for cell in cell_list:
        cell_type = determine_cell_type(cell)
        grn = grn_priors_array[cell_type]
        cell.grn = grn

        # Add the receptor nodes to the graph
        cell.grn.add_nodes_from(receptors[cell_type])

        # Add the ligand nodes to the graph
        cell.grn.add_nodes_from(ligands[cell_type])

    ###### Initilize the genes for the starting time points ######
    initial_t_lag = max(cfg.time_lags.values())
    for t in np.range(initial_t_lag):
        for cell in cell_list:
            cell.genes[t] = genes_trajectories[cell][t]

    ###### Datasets ######
    spatial_temporal_dataset = SpatialTemporalDataset(features_path, labels_path)

    train_dataset = spatial_temporal_dataset[: int(len(spatial_temporal_dataset) * 0.8)]
    test_dataset = spatial_temporal_dataset[int(len(spatial_temporal_dataset) * 0.8) :]

    graph_train_loader = geom_data.DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    graph_val_loader = geom_data.DataLoader(test_dataset, batch_size=cfg.data.batch_size)  # Additional loader for a larger datasets
    graph_test_loader = geom_data.DataLoader(test_dataset, batch_size=cfg.data.batch_size)
    


    ###### DATAMODULE ######

    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Data( x = Node feature matrix with shape [num_nodes, num_node_features],
    #       edge_index = Graph connectivity in COO format with shape [2, num_edges]., 
    #       edge_attr = Edge feature matrix with shape [num_edges, num_edge_features]. ,
    #       y = Graph-level or node-level ground-truth labels with arbitrary shape.
    # )

    ###### TRAINING ######
    pl.seed_everything(42)

    # Create a PyTorch Lightning trainer with the generation callback
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)

    trainer = pl.Trainer(
        default_root_dir=root_dir,
        callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
        accelerator="cuda",
        max_epochs=500,
        enable_progress_bar=False,
    )
    trainer.logger._default_hp_metric = None

    model = STAGEDModel(
        c_in=tu_dataset.num_node_features,
        c_out=1 if tu_dataset.num_classes == 2 else tu_dataset.num_classes,
        **model_kwargs,
    )

    trainer.fit(model, graph_train_loader, graph_val_loader)

    model = STAGEDModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation and test set
    train_result = trainer.test(model, dataloaders=graph_train_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=graph_test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "train": train_result[0]["test_acc"]}
    return model, result

# @hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
# def main(cfg: DictConfig) -> Optional[float]:
#     """Main entry point for training.

#     :param cfg: DictConfig configuration composed by Hydra.
#     :return: Optional[float] with optimized metric value.
#     """
#     # apply extra utilities
#     # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
#     extras(cfg)

#     # train the model
#     metric_dict, _ = train(cfg)

#     # safely retrieve metric value for hydra-based hyperparameter optimization
#     metric_value = get_metric_value(
#         metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
#     )

#     # return optimized metric
#     return metric_value

if __name__ == "__main__":
    main()


