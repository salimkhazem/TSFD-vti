import torch
import logging 
from tqdm.auto import tqdm 
from sklearn import metrics

logger_file = logging.getLogger('main_file')


def train_one_epoch(model, criterion, optimizer, data_loader, grad_norm, device):
    if device is None: 
        device = next(model.parameters()).device 
    loss_model, acc_model, total_f1_score = 0, 0, 0 
    model.train()
    with tqdm(data_loader, total=len(data_loader.dataset), desc=f"Training", leave=False) as pbar: 
        for batch in pbar:
            images, targets = batch["input"].to(device), batch["target"].to(device) 
            try: 
                assert images.shape[1] == model.in_channels, f"Expected {model.in_channels} channels, got {images.shape[0]}" 
            except: 
                assert images.shape[1] == model.module.in_channels, f"Expected {model.module.in_channels} channels, got {images.shape[0]}" 
            optimizer.zero_grad()
            output = model(images) 
            acc_model += (output.argmax(1) == targets).sum().item()
            y_true = targets.cpu().numpy().flatten()
            y_pred = output.argmax(1).cpu().numpy().flatten()
            batch_f1_score = metrics.f1_score(y_true, y_pred, average="macro", zero_division=1)
            total_f1_score += batch_f1_score * images.size(0)  # Multiply by batch size to weight the average
            loss = criterion(output, targets)
            loss_model += loss.item()
            loss.backward()
            if grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            optimizer.step()
            pbar.set_postfix(**{"train_loss": loss.item()})
            pbar.update(images.shape[0])
            del images, targets, output
            torch.cuda.empty_cache()
    return loss_model/len(data_loader.dataset), acc_model/len(data_loader.dataset), batch_f1_score, total_f1_score/len(data_loader.dataset)



@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    if device is None: 
        device = next(model.parameters()).device
    model.eval()
    total_loss = 0
    correct_pred = 0 
    total_f1_score = 0
    with tqdm(data_loader, total=len(data_loader.dataset), desc=f"Validation", leave=False) as pbar: 
        for batch in pbar:
            images, targets = batch["input"].to(device), batch["target"].to(device) 
            try: 
                assert images.shape[1] == model.in_channels, f"Expected {model.in_channels} channels, got {images.shape[0]}" 
            except: 
                assert images.shape[1] == model.module.in_channels, f"Expected {model.module.in_channels} channels, got {images.shape[0]}" 
            output = model(images)
            loss = criterion(output, targets)
            correct_pred += (output.argmax(1) == targets).sum().item()
            y_true = targets.cpu().numpy().flatten()
            y_pred = output.argmax(1).cpu().numpy().flatten()
            batch_f1_score = metrics.f1_score(y_true, y_pred, average="macro", zero_division=1)
            total_f1_score += batch_f1_score * images.size(0)  # Multiply by batch size to weight the average
            total_loss += loss.item()
            pbar.set_postfix(**{"eval_loss": loss.item()})
            pbar.update(images.shape[0])
            del images, targets, output
            torch.cuda.empty_cache()
    return total_loss/len(data_loader.dataset), correct_pred/len(data_loader.dataset), batch_f1_score, total_f1_score/len(data_loader.dataset)
