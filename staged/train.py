from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from tqdm import tqdm
import os
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # print("Raw Data Path:", cfg.data.raw_data_dir)
    # print("Processed Data Path:", cfg.data.processed_data_dir)
    print(OmegaConf.to_yaml(cfg))  # Print full config to verify interpolation

    # Define paths
    features_path = os.path.join(cfg.data.processed_data_dir,"features.csv")
    labels_path = os.path.join(cfg.data.processed_data_dir, "labels.csv")
    model_path = os.path.join(cfg.model.models_dir, "model.pkl")


    # Training logic
    logger.info(f"Training model with features: {features_path}, labels: {labels_path}, saving to {model_path}...")
    # metric_dict, _ = train(cfg)

    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    logger.success("Model training complete.")
    # return metric_value
    return


if __name__ == "__main__":
    main()

