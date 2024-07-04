import torch

from tqdm import tqdm 



def train_one_epoch(model, criterion, optimizer, data_loader, device):
    print("Training model")
    model.train()
    with tqdm(data_loader, total=len(data_loader), desc=f"Training", leave=True) as pbar: 
        total_loss = 0 
        for batch in data_loader:
            images, targets = batch["input"].to(device), batch["target"].to(device) 
            assert images.shape[1] == model.n_channels, f"Expected {model.n_channels} channels, got {images.shape[0]}" 
            optimizer.zero_grad()
            output = model(images)
            #print(output.shape, targets.shape)
            loss = criterion(output, targets)
            pbar.set_postfix(**{"train_loss": loss.item()})
            loss.backward()
            optimizer.step()
            pbar.update(images.shape[0])
            total_loss += loss.item()
            #print(f"Shape of output: {output.shape}")
            pbar.set_postfix({"loss": total_loss / len(data_loader)})



