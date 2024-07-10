import torch
import logging 
from tqdm.auto import tqdm 

logger_file = logging.getLogger('main_file')


def train_one_epoch(model, criterion, optimizer, data_loader, grad_norm, device):
    if device is None: 
        device = next(model.parameters()).device 
    loss_model, acc_model = 0, 0 
    model.train()
    with tqdm(data_loader, desc=f"Training", leave=False) as pbar: 
        for batch in pbar:
            images, targets = batch["input"].to(device), batch["target"].to(device) 
            try: 
                assert images.shape[1] == model.in_channels, f"Expected {model.in_channels} channels, got {images.shape[0]}" 
            except: 
                assert images.shape[1] == model.module.in_channels, f"Expected {model.module.in_channels} channels, got {images.shape[0]}" 
            optimizer.zero_grad()
            output = model(images) 
            acc_model += (output.argmax(1) == targets).sum().item()
            loss = criterion(output, targets)
            loss_model += loss.item()
            loss.backward()
            if grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
            pbar.set_postfix(**{"train_loss": loss.item()})
            pbar.update(images.shape[0])
    return loss_model/len(data_loader.dataset), acc_model/len(data_loader.dataset)



@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    if device is None: 
        device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    correct_pred = 0 
    with tqdm(data_loader, desc=f"Validation", leave=False) as pbar: 
        for batch in pbar:
            images, targets = batch["input"].to(device), batch["target"].to(device) 
            assert images.shape[1] == model.in_channels, f"Expected {model.in_channels} channels, got {images.shape[0]}" 
            output = model(images)
            loss = criterion(output, targets)
            correct_pred += (output.argmax(1) == targets).sum().item()
            total_loss += loss.item()
            pbar.set_postfix(**{"eval_loss": loss.item()})
            pbar.update(images.shape[0])
    return total_loss/len(data_loader.dataset), correct_pred/len(data_loader.dataset)
