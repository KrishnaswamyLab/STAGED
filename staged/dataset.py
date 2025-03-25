from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):

    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path = os.path.join(cfg.data.raw_data_dir,"dataset.csv")
    output_path = os.path.join(cfg.data.processed_data_dir,"dataset.csv")
    # ----------------------------------------------

    
    logger.info("Processing dataset...")

    ## Creating spatial graphs from spatial data

    ## Retrieving cell trajectories from MIOFlow

    ## Retrieving GRN from RITINI

    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    main()
