import torch
import wandb
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from strnn.models.strNN import StrNN
from torch.optim import Optimizer
from torch.utils.data import DataLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data_and_adj_mtx(dataset_name, adj_mtx_name):
    """
    Load train and val splits of specified data and
    associated adjacency matrix
    """
    # Load adjacency matrix
    if adj_mtx_name != "None":
        adj_mtx_path = f"./synth_data_files/{adj_mtx_name}.npz"
        adj_mtx = np.load(adj_mtx_path)
        adj_mtx = adj_mtx[adj_mtx.files[0]]
    else:
        adj_mtx = None

    # Load data
    data_path = f"./synth_data_files/{dataset_name}.npz"
    data = np.load(data_path)
    train_data = data['train_data'].astype(np.float32)
    val_data = data['valid_data'].astype(np.float32)
    train_data = torch.from_numpy(train_data).to(device)
    val_data = torch.from_numpy(val_data).to(device)

    return train_data, val_data, adj_mtx


# def train_loop(
#     model: StrNN,
#     optimizer: Optimizer,
#     train_dl: DataLoader,
#     val_dl: DataLoader,
#     max_epoch: int,
#     patience: int
# ) -> dict | None:
#     """
#     @param model
#     @param optimizer
#     @param train_dl: training data loader
#     @param val_dl: validation data loader
#     @param max_epoch: maximum epoch allowed for training
#     @param patience: max number of epochs without validation loss
#             improvement before terminating
#     @return: dict of best model state
#     """
#     best_model_state = None
#     best_val = None
#     counter = 0
#
#     for epoch in range(1, max_epoch):
#         train_losses = []
#         val_losses = []
#
#         for batch in train_dl:
#             # Training step
#             optimizer.zero_grad()
#             x = batch
#             x_hat, loss = model.get_preds_loss(x)
#             train_loss = loss.item()
#
#             loss.backward()
#             optimizer.step()
#
#             train_losses.append(train_loss)
#
#         # Validation step
#         with torch.no_grad():
#             for batch in val_dl:
#                 x = batch
#                 x_hat, loss = model.get_preds_loss(x)
#                 val_loss = loss.item()
#                 val_losses.append(val_loss)
#
#             epoch_train_loss = np.mean(train_losses)
#             epoch_val_loss = np.mean(val_losses)
#
#             # TODO: Add scheduler
#
#             # wandb logging
#             wandb.log({
#                 "train_loss": epoch_train_loss,
#                 "val_loss": epoch_val_loss
#             })
#
#             if best_val is None or epoch_val_loss < best_val:
#                 best_val = epoch_val_loss
#                 best_model_state = model.state_dict()
#                 counter = 0 # Reset counter since we beat best loss so far
#             else:
#                 counter += 1
#                 if counter > patience:
#                     return best_model_state
#
#     return best_model_state


def train_loop(
    model,
    optimizer,
    train_dl,
    val_dl,
    max_epoch,
    patience
):
    best_model_state = None
    best_val_loss = float('inf')
    counter = 0

    # Initialize lists to track per-epoch loss
    train_losses_per_epoch = []
    val_losses_per_epoch = []

    for epoch in range(1, max_epoch + 1):
        train_losses = []
        val_losses = []

        for batch in train_dl:
            if isinstance(batch, torch.Tensor):
                x = batch  # features are the entire batch
                optimizer.zero_grad()
                x_hat, loss = model.get_preds_loss(x)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

        with torch.no_grad():
            for x in val_dl:
                x_hat, loss = model.get_preds_loss(x)
                val_losses.append(loss.item())

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        train_losses_per_epoch.append(epoch_train_loss)
        val_losses_per_epoch.append(epoch_val_loss)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss
        })

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    # Return all the metrics collected during training
    return {
        "best_model_state": best_model_state,
        "train_losses_per_epoch": train_losses_per_epoch,
        "val_losses_per_epoch": val_losses_per_epoch
    }
