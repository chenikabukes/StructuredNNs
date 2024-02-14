import argparse
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
import re
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from binary_gaussian_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator
from strnn.models.strNN import MaskedLinear
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("Runs StrNN on synthetic dataset.")
parser.add_argument("--experiment_name", type=str, default="multimodal")
parser.add_argument("--data_seed", type=int, default=2547)
parser.add_argument("--scheduler", type=str, default="plateau")
parser.add_argument("--model_seed", type=int, default=2647)
parser.add_argument("--wandb_name", type=str)

args = parser.parse_args()

# TODO
def main():
    with open("./experiment_config.yaml", "r") as f:
        configs = yaml.safe_load(f)
    experiment_config = configs[args.experiment_name]

    dataset_name = experiment_config["dataset_name"]
    adj_mtx_name = experiment_config["adj_mtx_name"]
    train_data, val_data, adj_mtx = load_data_and_adj_mtx(dataset_name, adj_mtx_name)
    input_size = len(train_data[0])
    experiment_config["input_size"] = input_size

    batch_size = experiment_config["batch_size"]
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    hidden_size_mults = [experiment_config[f"hidden_size_multiplier_{i}"] for i in range(1, 6)]

    if "binary" in dataset_name:
        data_type = "binary"
        output_size = input_size
    elif "gaussian" in dataset_name:
        data_type = "gaussian"
        output_size = 2 * input_size
    else:
        raise ValueError("Data type must be binary or Gaussian!")

    run = wandb.init(project=args.wandb_name, config=experiment_config, reinit=True)
    overall_variances = {}

    for num_layers in range(1, 10):
        hidden_sizes = [h * input_size for h in hidden_size_mults[:num_layers]]

        model = StrNNDensityEstimator(
            nin=input_size,
            hidden_sizes=hidden_sizes,
            nout=output_size,
            opt_type=experiment_config["opt_type"],
            opt_args={},
            precomputed_masks=None,
            adjacency=adj_mtx,
            activation=experiment_config["activation"],
            data_type=data_type
        )
        model.to(device)

        optimizer = AdamW(
            model.parameters(),
            lr=experiment_config["learning_rate"],
            eps=experiment_config["epsilon"],
            weight_decay=experiment_config["weight_decay"]
        )

        results = train_loop(
            model,
            optimizer,
            train_dl,
            val_dl,
            experiment_config["max_epochs"],
            experiment_config["patience"]
        )
        total_variance = 0
        total_elements = 0

        for idx, layer in enumerate(model.net_list):
            if isinstance(layer, MaskedLinear):
                layer_variance = torch.var(layer.weight.data * layer.mask).item()
                total_variance += layer_variance * layer.mask.numel()
                total_elements += layer.mask.numel()  # total num elements in the tensor

        # Calculate the overall variance for the current model configuration (with x num_layers)
        if total_elements > 0:
            overall_variance = total_variance / total_elements
            overall_variances[num_layers] = overall_variance

    variance_table = wandb.Table(columns=['Model Configuration', 'Overall Weight Variance'])
    for num_layers, variance in overall_variances.items():
        variance_table.add_data(f"{num_layers} Layers", variance)
    wandb.log({"Overall Variance by Num Layers": variance_table})

    wandb.finish()


if __name__ == "__main__":
    main()
