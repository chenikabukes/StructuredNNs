import argparse
import yaml
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from binary_gaussian_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("Runs StrNN on synthetic dataset.")
parser.add_argument("--experiment_name", type=str, default="multimodal")
parser.add_argument("--data_seed", type=int, default=2547)
parser.add_argument("--scheduler", type=str, default="plateau")
parser.add_argument("--model_seed", type=int, default=2647)
parser.add_argument("--wandb_name", type=str)

args = parser.parse_args()


# def main():
#     # Load experiment configs
#     with open("./experiment_config.yaml", "r") as f:
#         configs = yaml.safe_load(f)
#         experiment_config = configs[args.experiment_name]
#
#     # Load data
#     dataset_name = experiment_config["dataset_name"]
#     adj_mtx_name = experiment_config["adj_mtx_name"]
#     train_data, val_data, adj_mtx = load_data_and_adj_mtx(
#         dataset_name, adj_mtx_name
#     )
#     input_size = len(train_data[0])
#     experiment_config["input_size"] = input_size
#
#     # Specify hidden layer sizes
#     num_hidden_layers = experiment_config["num_hidden_layers"]
#     hidden_size_mults = []
#     for i in range(1, 6):
#         hidden_size_mults.append(
#             experiment_config[f"hidden_size_multiplier_{i}"]
#         )
#     hidden_sizes = [h * input_size for h in hidden_size_mults]
#     hidden_sizes = tuple(hidden_sizes[:num_hidden_layers])
#     assert isinstance(hidden_sizes[0], int)
#
#     # Make dataloaders
#     batch_size = experiment_config["batch_size"]
#     train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
#
#     # Intialize model (fix random seed if necessary)
#     if args.model_seed is not None:
#         np.random.seed(args.model_seed)
#         torch.random.manual_seed(args.model_seed)
#
#     if "binary" in dataset_name:
#         data_type = "binary"
#         output_size = input_size
#     elif "gaussian" in dataset_name:
#         data_type = "gaussian"
#         output_size = 2 * input_size
#     else:
#         raise ValueError("Data type must be binary or Gaussian!")
#     # Question: why initialisation even matters in first place?
#     Goal we want hidden size 5+
#     Manually imbalance variances
#     Set the variance to something non-sensical with default init
#
#     Also track training curves
#     Graph sparsity vs test performance
#     Graph depth vs test performance
#     model = StrNNDensityEstimator(
#         nin=input_size,
#         hidden_sizes=hidden_sizes,  # at what depth and what sparsity does initialisation impact accuracy
#         nout=output_size,
#         opt_type=experiment_config["opt_type"],
#         opt_args={},
#         precomputed_masks=None,
#         adjacency=adj_mtx,
#         activation=experiment_config["activation"],
#         data_type=data_type
#     )
#     model.to(device)
#
#     # Initialize optimizer
#     optimizer_name = experiment_config["optimizer"]
#     if optimizer_name == "adamw":
#         optimizer = AdamW(
#             model.parameters(),
#             lr=experiment_config["learning_rate"],
#             eps=experiment_config["epsilon"],
#             weight_decay=experiment_config["weight_decay"]
#         )
#     else:
#         raise ValueError(f"{optimizer_name} is not a valid optimizer!")
#
#     run = wandb.init(
#         project=args.wandb_name,
#         config=experiment_config
#     )
#
#     best_model_state = train_loop(
#         model,
#         optimizer,
#         train_dl,
#         val_dl,
#         experiment_config["max_epochs"],
#         experiment_config["patience"]
#     )

# def main():
#     with open("./experiment_config.yaml", "r") as f:
#         configs = yaml.safe_load(f)
#     experiment_config = configs[args.experiment_name]
#
#     dataset_name = experiment_config["dataset_name"]
#     adj_mtx_name = experiment_config["adj_mtx_name"]
#     train_data, val_data, adj_mtx = load_data_and_adj_mtx(dataset_name, adj_mtx_name)
#     input_size = len(train_data[0])
#     experiment_config["input_size"] = input_size
#
#     batch_size = experiment_config["batch_size"]
#     train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)
#
#     hidden_size_mults = [experiment_config[f"hidden_size_multiplier_{i}"] for i in range(1, 6)]
#
#     if "binary" in dataset_name:
#         data_type = "binary"
#         output_size = input_size
#     elif "gaussian" in dataset_name:
#         data_type = "gaussian"
#         output_size = 2 * input_size
#     else:
#         raise ValueError("Data type must be binary or Gaussian!")
#
#     run = wandb.init(
#         project=args.wandb_name,
#         config=experiment_config,
#         reinit=True
#     )
#     final_val_losses = []
#
#     for num_layers in range(1, 10):
#         hidden_sizes = tuple([h * input_size for h in hidden_size_mults][:num_layers])
#
#         model = StrNNDensityEstimator(
#             nin=input_size,
#             hidden_sizes=hidden_sizes,
#             nout=output_size,
#             opt_type=experiment_config["opt_type"],
#             opt_args={},
#             precomputed_masks=None,
#             adjacency=adj_mtx,
#             activation=experiment_config["activation"],
#             data_type=data_type
#         )
#         model.to(device)
#
#
#
#         optimizer = AdamW(
#             model.parameters(),
#             lr=experiment_config["learning_rate"],
#             eps=experiment_config["epsilon"],
#             weight_decay=experiment_config["weight_decay"]
#         )
#
#         results = train_loop(
#             model,
#             optimizer,
#             train_dl,
#             val_dl,
#             experiment_config["max_epochs"],
#             experiment_config["patience"]
#         )
#
#         # vanilla strnn and use super sparse, use Kaiming to initialise the weights and then apply mask and calculate the variances.
#         # - to understand what a normal range of variance will be in very sparse network (should be 1/n for each node)
#         # how much less than 1.
#
#         run = wandb.run
#
#         # Log training and validation losses to wandb
#         for epoch in range(len(results['train_losses_per_epoch'])):
#             wandb.log({
#                 'Epoch': epoch,
#                 f'Train Loss {num_layers} Layers': results['train_losses_per_epoch'][epoch],
#                 f'Validation Loss {num_layers} Layers': results['val_losses_per_epoch'][epoch]
#             })
#
#     wandb.finish()
#     run.finish()


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

    data_type = "binary" if "binary" in dataset_name else "gaussian"
    output_size = input_size if data_type == "binary" else 2 * input_size

    run = wandb.init(project=args.wandb_name, config=experiment_config, reinit=True)

    final_val_losses = []
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

        # train/validation losses
        fig, ax = plt.subplots()
        ax.plot(range(len(results['train_losses_per_epoch'])), results['train_losses_per_epoch'], label='Train Loss')
        ax.plot(range(len(results['val_losses_per_epoch'])), results['val_losses_per_epoch'], label='Validation Loss')
        ax.set_title(f'Train/Validation Loss for {num_layers} Layers')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        wandb.log({f'Train/Validation Loss for {num_layers} Layers': wandb.Image(fig)})
        plt.close(fig)

        final_val_losses.append(results['val_losses_per_epoch'][-1])

    # final validation losses over layers
    fig, ax = plt.subplots()
    ax.plot(range(1, 10), final_val_losses, marker='o', linestyle='-')
    ax.set_title('Final Validation Loss Over Layers')
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Final Validation Loss')
    wandb.log({'Final Validation Loss Over Layers': wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    main()
