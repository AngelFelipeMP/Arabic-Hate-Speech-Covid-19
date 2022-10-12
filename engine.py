import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    return nn.BCELoss() (outputs, targets.view(-1,1))


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    fin_targets = []
    fin_predictions = []
    total_loss = 0
    
    
    for batch in data_loader:
        batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
        targets = batch["targets"].type(torch.FloatTensor)
        del batch["targets"]
        
        optimizer.zero_grad()
        outputs = model(batch)
        loss = loss_fn(outputs, targets)
        
        total_loss += loss.cpu().detach().numpy().tolist()
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_predictions.extend(outputs.cpu().detach().numpy().tolist())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    return fin_predictions, fin_targets, total_loss/len(data_loader)
        
        


def eval_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_predictions = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in data_loader:
            
            batch = {k:v.to(device, dtype=torch.long) for k,v in batch.items()}
            targets = batch["targets"].type(torch.FloatTensor)
            del batch["targets"]

            outputs = model(batch)
            loss = loss_fn(outputs, targets)
            total_loss += loss.cpu().detach().numpy().tolist()
            
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_predictions.extend(outputs.cpu().detach().numpy().tolist())
    
    return fin_predictions, fin_targets, total_loss/len(data_loader)