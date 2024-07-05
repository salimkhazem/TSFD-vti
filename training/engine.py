import torch

from tqdm import tqdm 



def train_one_epoch(model, criterion, optimizer, data_loader, device):
    print("Training model")
    model.train()
    with tqdm(data_loader, total=len(data_loader), desc=f"Training", leave=True) as pbar: 
        total_loss = 0 
        for batch in pbar:
            images, targets = batch["input"].to(device), batch["target"].to(device) 
            assert images.shape[1] == model.n_channels, f"Expected {model.n_channels} channels, got {images.shape[0]}" 
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, targets)
            pbar.set_postfix(**{"train_loss": loss.item()})
            loss.backward()
            optimizer.step()
            pbar.update(images.shape[0])
            total_loss += loss.item()
            pbar.set_postfix({"loss": total_loss / len(data_loader)})




@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    if device is None: 
        device = next(model.parameters()).device
    model.eval()
    with tqdm(data_loader, total=len(data_loader), desc=f"Evaluating", leave=False) as pbar: 
        total_loss = 0
        correct_pred = 0 
        for batch in pbar:
            images, targets = batch["input"].to(device), batch["target"].to(device) 
            assert images.shape[1] == model.n_channels, f"Expected {model.n_channels} channels, got {images.shape[0]}" 
            output = model(images)
            loss = criterion(output, targets)
            correct_pred += (output.argmax(1) == targets).sum().item()
            total_loss += loss.item()
            pbar.set_postfix(**{"eval_loss": loss.item()})
            pbar.update(images.shape[0])
            pbar.set_postfix({"loss": total_loss / len(data_loader)})
