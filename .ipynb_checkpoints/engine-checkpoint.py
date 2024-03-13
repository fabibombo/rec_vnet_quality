import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from typing import Dict, List, Tuple
import gc

import pandas as pd
from tqdm.auto import tqdm

import utils

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    criterion: torch.nn.Module, 
    device: torch.device,
) -> Dict[str, float]:
    
    model.train()

    train_loss = 0
    train_metrics = {"mae": 0}
    
    # torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

    for batch, (X, y) in enumerate(dataloader):
        #X, y = X.to(device), y.to(device)

        y_out = model(X.to(device)).to("cpu")

        # if torch.any(torch.isnan(y_out)):
        #     print("Batch:", batch)
        #     print("NaN alert!")
        #     print(torch.isnan(X).any(), torch.isnan(y).any(), torch.isnan(y_out).any())
        #     torch.save({"X": X, "y": y, "preds": y_out}, "troubleshoot_nans.pt")
        
        loss = criterion(y_out, y.view(-1))

        train_loss += loss.detach().item()

        optimizer.zero_grad()
        
        loss.backward(retain_graph=None)

        # add gradient clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
        
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        #TODO
        metric_mae = F.l1_loss(y_out, y.view(-1), reduction = 'sum').detach().item()
        
        train_metrics["mae"] += metric_mae
        
        #cleaning memory
        gc.collect()
        torch.cuda.empty_cache()
            
        ## print every 20%
        if batch != 0 and batch % (len(dataloader) // 5) == 0 and batch < (9 * (len(dataloader) // 10)):
            print(
                "\t",
                batch,
                "\t",
                "Loss:",
                round(train_loss / (batch + 1), 4),
                "MAE:",
                round(train_metrics["mae"] / (batch + 1), 4)
            )

    for key in train_metrics:
        train_metrics[key] /= len(dataloader)

    return {"loss": train_loss / len(dataloader), **train_metrics}

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module, 
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    test_loss = 0
    test_metrics = {"mae": 0}

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            #X, y = X.to(device), y.to(device)
    
            # flatten y for nll
            # y = y.view(-1)
            y_out = model(X.to(device)).to("cpu")

            loss = criterion(y_out, y.view(-1)).detach().item()
                
            test_loss += loss

            #TODO
            metric_mae = F.l1_loss(y_out, y.view(-1), reduction = 'sum').detach().item()
            
            test_metrics["mae"] += metric_mae

            gc.collect()
            torch.cuda.empty_cache()
        
        test_loss /= len(dataloader)
        for key in test_metrics:
            test_metrics[key] /= len(dataloader)

    return {"loss": test_loss, **test_metrics}

def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    criterion: torch.nn.Module, 
    epochs: int,
    results_path: str,
    models_path: str,
    device: torch.device,
) -> Dict[str, List]:
    
    results = {
        "epoch": [],
        "train_loss": [],
        "train_mae": [],
        "test_loss": [],
        "test_mae": []
    }

    min_test_metric = None
    
    print(" ## BEGIN TRAINING ## ")
    print("    Model:                 \t", model.name)
    print("    Number of train batches:\t", len(train_dataloader))
    print("    Number of test batches:\t", len(test_dataloader))

    for epoch in tqdm(range(epochs)):
        train_results = train_step(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            criterion=criterion,
            device=device
        )
        test_results = test_step(
            model=model, 
            criterion=criterion,
            dataloader=test_dataloader, 
            device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_results['loss']:.4f} | "
            f"train_bcloss: {train_results['mae']:.4f} | "
            f"test_loss: {test_results['loss']:.4f} |  "
            f"test_bcloss: {test_results['mae']:.4f}"
        )

        results["epoch"].append(epoch)
        results["train_loss"].append(train_results["loss"])
        results["train_bcloss"].append(train_results["mae"])
        results["test_loss"].append(test_results["loss"])
        results["test_bcloss"].append(test_results["mae"])

        write_results(results, results_path +"results_" + model.name + ".csv")

        if min_test_metric is None:
            min_test_metric = test_results["mae"]
        elif epoch > 20 and test_results["mae"] > min_test_metric:
            min_test_metric = test_results["mae"]
            utils.save_model(model=model,
                           target_dir=models_path,
                           model_name=model.name + "_weights_min_test_metric.pth")

        #save the last trained epoch with overwritting in case the training stops and we need to resume it
        utils.save_model(model=model,
                           target_dir=models_path,
                           model_name=model.name + "_last_trained_epoch.pth")
    return results

def write_results(results, csv_path):
    # we send everything to cpu just in case
    results_cpu = {}
    for key, value in results.items():
        tmp_list = []
        for item in results[key]:
            if torch.is_tensor(item):
                tmp_list.append(item.cpu().detach().numpy())
            else:
                tmp_list.append(item)
            results_cpu[key] = tmp_list

    df = pd.DataFrame(results_cpu)
    df.to_csv(csv_path, index=False)
