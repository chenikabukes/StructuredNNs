import torch
import wandb
import numpy as np
import sys
import os
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(dataset_name='CiteSeer', transform=NormalizeFeatures()):
    """
    Load the CiteSeer dataset and preprocess it.
    """
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name, transform=transform)
    data = dataset[0].to(device)  # Move data to the appropriate device
    num_features = dataset.num_node_features
    num_classes = dataset.num_classes

    # In the Planetoid dataset, data is already split into train, val, and test
    train_data = data[data.train_mask]
    val_data = data[data.val_mask]
    test_data = data[data.test_mask]

    return train_data, val_data, test_data, num_features, num_classes


def load_data(dataset_name='CiteSeer', transform=NormalizeFeatures()):
    """
    Load the CiteSeer dataset, preprocess it, and create an adjacency matrix.
    """
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name, transform=transform)
    data = dataset[0].to(device)

    # Create an adjacency matrix from edge indices
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=device)
    adj_matrix[edge_index[0], edge_index[1]] = 1

    num_features = dataset.num_node_features
    num_classes = dataset.num_classes

    return data, adj_matrix, num_features, num_classes


def train_model(model, train_data, optimizer, criterion):
    """
    Train the model for one epoch.
    """
    model.train()
    optimizer.zero_grad()
    out = model(train_data.x)
    loss = criterion(out[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_model(model, data, criterion):
    """
    Evaluate the model on validation or test data.
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x)
        loss = criterion(out[data.val_mask], data.y[data.val_mask]).item()  # Change to data.test_mask for test evaluation
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = int(pred[data.val_mask].eq(data.y[data.val_mask]).sum().item())  # Change to data.test_mask for test evaluation
        accuracy = correct / int(data.val_mask.sum())  # Change to data.test_mask for test evaluation
    return loss, accuracy
