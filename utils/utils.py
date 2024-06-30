import os
import random
import torch
import numpy as np


def get_optimizer(cfg, model):
    optimizer_cls = cfg["Optimizer"]["Name"]
    optimizer_args = {k: v for k, v in cfg["Optimizer"].items() if k != "Name"}
    return eval(f"torch.optim.{optimizer_cls}")(
        model.parameters(), **optimizer_args
    )


def get_criterion(cfg):
    criterion_cls = cfg["Loss"]["Name"]
    criterion_args = {k: v for k, v in cfg["Loss"].items() if k != "Name"}
    return eval(f"torch.nn.{criterion_cls}")(**criterion_args)


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = logdir / run_name
        if not log_path.exists():
            log_path.mkdir(parents=True)
            return log_path
        i = i + 1


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    import sys
    import yaml
    import pathlib

    with open(sys.argv[1]) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    print(cfg)
    optimizer = get_optimizer(cfg, torch.nn.Linear(1, 1))
    criterion = get_criterion(cfg)
    logdir = pathlib.Path("logs")
    logdir = generate_unique_logpath(logdir, "experiment_1")
    print(logdir)
    seed_everything(42)
