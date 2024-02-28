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

    variance_scales = [0.01,0.05, 0.1, 0.5, 1]
    val_losses = []

    num_layers = experiment_config["num_hidden_layers"]
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
        data_type=data_type,
    )
    model.to(device)

    for scale in variance_scales:
        for idx, layer in enumerate(model.net_list):
            if isinstance(layer, MaskedLinear):
                layer.reset_parameters_w_masking(scale=scale)

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
        final_val_loss = results['val_losses_per_epoch'][-1]
        val_losses.append(final_val_loss)

    fig, ax = plt.subplots()
    ax.plot(variance_scales, val_losses, marker='o', linestyle='-', color='b')
    ax.set_title('Validation Loss over Scalar Multipliers of Initial Variance')
    ax.set_xlabel('Scalar Multiplier of Initial Variance')
    ax.set_ylabel('Final Validation Loss')
    plt.grid(False)
    plt.tight_layout()

    # Log the plot to Weights & Biases
    plot_filename = 'variance_scale_vs_val_loss.png'
    plt.savefig(plot_filename)
    plt.close()

    wandb.log({"Validation Loss over Scalar Multipliers of Initial Variance": wandb.Image(plot_filename)})

    # End the Weights & Biases run
    wandb.finish()


if __name__ == "__main__":
    main()
