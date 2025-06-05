from pathlib import Path
import os

from loguru import logger
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):

    features_path = os.path.join(cfg.data.processed_data_dir, "test_features.csv")
    model_path = os.path.join(cfg.model.model_dir, "model.pkl")
    predictions_path = os.path.join(cfg.reports.predictions_dir, "test_predictions.csv")
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Performing inference for model...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Inference complete.")
    # -----------------------------------------


if __name__ == "__main__":
    main()
