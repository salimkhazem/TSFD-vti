import argparse
import shutil
import copy
import time 
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torchinfo import summary
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
    model.to(device)
    return model

def init_optimizer_loss(config, model, logger): 
    if config["Loss"]["Name"] is None: 
        loss = nn.CrossEntropyLoss()
    else: 
        loss = utils.get_criterion(config)
    optimzer = utils.get_optimizer(config, model)
    logger.info(f"Loss function:\n{loss}")
    logger.info(f"Optimizer:\n{optimzer}")
    return loss, optimzer

def train(config, model, criterion, optimizer, train_loader, valid_loader, epochs, device, logger, path_logs): 
    logger.info(f"{str('-'*5)} Training the model for {epochs} epochs {str('-'*5)}") 
    grad_norm = config["Training"]["Clip_grad_norm"]
    best_acc = 0 
    for epoch in tqdm(range(1, epochs+1), desc=f'Training loop'): 
        loss_epoch, acc_epoch = engine.train_one_epoch(model, criterion, optimizer, 
                                                       train_loader, grad_norm=grad_norm, device=device) 
        logger.info(f"Epoch: {epoch} - Loss: {loss_epoch:.4f} - Accuracy: {acc_epoch:.4f}") 

        loss_val, acc_val = engine.evaluate(model, criterion, valid_loader, device) 
        logger.info(f"Validation - Loss: {loss_val:.4f} - Accuracy: {acc_val:.4f}")
        if acc_val > best_acc: 
            best_acc = acc_val 
            logger.info(f"New best accuracy: {best_acc:.4f} .... Saving model")
            torch.save(model.state_dict(), path_logs / "best_model.pth")


def resume(config, model, path_logs, logger): 
    if config["Training"]["Checkpoint"] is not None:
        resume_training = config["Training"]["Checkpoint"] 
    else: 
        config["Training"]["Checkpoint"]["Resume"] = False
        resume_training = config["Training"]["Checkpoint"]

    if resume_training["Resume"]: 
        path_previous_logs = (Path(__file__).resolve().parent[2].absolute() 
                              / "logs" / str(resume_training["Date"])
                              / resume_training["Time"])
        
        if path_previous_logs.exists():
            resume_training["Path"] = path_previous_logs 
        else: 
            logger.warning("Folder: {path_previous_logs} does not exist"
                           " .... Checkpoint aborted")
            resume_training["Resume"] = False
    return resume_training


def track_experiments(fct): 
    def inner(logger, *args, **kwargs): 
        output = None 
        # CO2 tracker 
        try: 
            tracker = EmissionsTracker(log_level='error',
                                       logging_logger=logger) 
            tracker.start()
        except Exception as e:
            tracker = None 
            logger.warning(f"CO2 tracker failed: {e}") 
        
        # Time tracker
        start_time = time.time() 

        try: 
            output = fct(logger, *args, **kwargs) 
        
        except Exception as error:
            logger.exception(f"Error in main:\n {error}")
        finally:
            total_time_s = int(time.time() - start_time) 
            h, remainder = divmod(total_time_s, 3600) 
            mins, secs = divmod(remainder, 60) 
            total_time = f"{h:02d}H:{mins:02d}min:{secs:02}s"

            emission = tracker.stop() if tracker is not None else None
            if emission is None: 
                emission = 0. 
                logger.warning("CO2 tracker failed, there is no value for emission")
        
        logger.info(f'Total time: {total_time}')
        logger.info(f'CO2 emissions: {emission:.3e} kg')

        return output 
    return inner


@track_experiments
def main(logger, config, device, path_logs, train_data, valid_data, is_train, is_test): 
    batch_size = config["Data"]["Batch_size"]
    epochs = config["Training"]["Epochs"]
    shuffle = config["Data"]["Shuffle"]
    # Initialize data
    train_loader, valid_loader = data_loader.create_dataloaders(train_data, valid_data, batch_size, shuffle)
    # Initialize model
    model = init_model(config, device, logger) 

    # Initialize optimizer and loss
    loss, optimizer = init_optimizer_loss(config, model, logger)

    # Resume training
    resume_training = resume(config, model, path_logs, logger)
    
    # Example log to demonstrate functionality
    logger.info("Initialization complete.")

    # Train model
    if is_train: 
        logger.info(f"{str('-'*20)} Training {str('-'*20)}") 
        train(config, model, loss, optimizer, train_loader, valid_loader, epochs, device, logger, path_logs)

    # Test model
    if is_test: 
        pass 

if __name__ == "__main__":
    parser = argparse.ArgumentParser('main') 
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    # Initialize configuration and logger
    config, logger, path_logs = init_config(args.config)

    device = utils.get_device()

    # Initialize data
    train_dataset, valid_dataset = data_init(config, logger, device)

    # Run main function
    main(logger, config, device, path_logs, train_dataset, valid_dataset, is_train=True, is_test=False)


   
