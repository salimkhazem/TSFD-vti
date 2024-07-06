import argparse
import shutil
import copy
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torchinfo import summary
import numpy as np
from functools import partial
from codecarbon import EmissionsTracker
from . import init_logger
from . import models
from .data import data_loader
from .utils import utils
from .training import engine


def init_config(cfg):
    with open(cfg, "r") as f:
        config = yaml.safe_load(f)
    logs_kwargs = dict()
    logger, _, _, path_logs = init_logger(
        path_logs="./pipeline_vti/logs", **logs_kwargs
    )
    print(path_logs)
    with open(path_logs / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(
        f" Path to config file: {Path(cfg)}\n Config file is copied to the logs dir"
        f" {path_logs}/config.yaml\n"
        f" Used Configuration:\n {yaml.dump(config)}"
    )

    return config, logger, path_logs


def data_init(config, logger, device=None):
    random_seed = config["Training"]["SEED"]
    logger.info(f"Using random seed: {random_seed}")
    utils.seed_everything(random_seed)
    train_dataset, valid_dataset, train_paths, valid_paths = (
        data_loader.create_datasets(
            config["Dataset"]["args"]["root_dir"],
            validation_split=config["Dataset"]["args"]["validation_split"],
        )
    )
    logger.info(
        f"Samples per set: Training= {len(train_dataset)}"
        f" - Validation= {len(valid_dataset)}"
    )
    return train_dataset, valid_dataset


def init_model(config, device, logger):
    model_cls = config["Model"]["Name"]
    model_args = config["Model"]["args"]
    input_shape = (
        config["Dataset"]["args"]["resize"],
        config["Dataset"]["args"]["resize"],
    )
    if model_args is not None:
        model = eval(f"models.{model_cls}(**model_args)")
    try:
        model_summary = summary(model, input_size=(1, input_shape), verbose=0)
        logger.info(f"Model summary:\n{str(model_summary)}")
    except Exception as e:
        logger.warn(f"Model summary failed using torchinfo: {e}")
    logger.info(f"Model print:\n{model}")

    return model


if __name__ == "__main__":
    config_file = "/home/GPU/skhazem/VTI_exp/pipeline_vti/config/Contours.yml"

    # Initialize configuration and logger
    config, logger, path_logs = init_config(config_file)

    device = utils.get_device()

    # Initialize data
    train_dataset, valid_dataset = data_init(config, logger, device)

    # Initialize model
    model = init_model(config, device, logger)
    # Example log to demonstrate functionality
    logger.info("Initialization complete.")
