import sys 
import yaml 
import torch 
import pathlib 
from utils import utils
from models import model 
from data import data_loader
from training import engine 

with open(sys.argv[1]) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using device {device}: model will be moved to this device")
model = model.UNet(n_channels=1, n_classes=1).to(device)
optimizer = utils.get_optimizer(cfg, model)
criterion = utils.get_criterion(cfg)
train_dataset, valid_dataset, train_paths, valid_paths = data_loader.create_datasets(
        cfg["Dataset"]["args"]["root_dir"], validation_split=0.2
    )
train_loader, valid_loader = data_loader.create_dataloaders(
        train_dataset, valid_dataset, batch_size=8, 
    )
logdir = pathlib.Path("logs")
logdir = utils.generate_unique_logpath(logdir, "experiment_1")
engine.train_one_epoch(model, criterion, optimizer, valid_loader, device)
engine.evaluate(model, criterion, valid_loader, device)


