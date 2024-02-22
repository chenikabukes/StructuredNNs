import argparse
import yaml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt

from gradient_activation_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser("Runs StrNN on synthetic dataset.")
parser.add_argument("--experiment_name", type=str, default="default_experiment")
parser.add_argument("--wandb_name", type=str, default="default_project")

args = parser.parse_args()


def main():
    with open("./experiment_config.yaml", "r") as f:
        configs = yaml.safe_load(f)
    experiment_config = configs[args.experiment_name]

    # Load dataset and adjacency matrix
    dataset_name = experiment_config["dataset_name"]
    adj_mtx_name = experiment_config["adj_mtx_name"]
    train_data, val_data, adj_mtx = load_data_and_adj_mtx(
        dataset_name, adj_mtx_name
    )
    input_size = train_data.shape[1]

    train_dl = DataLoader(train_data, batch_size=experiment_config["batch_size"], shuffle=True)
    val_dl = DataLoader(val_data, batch_size=experiment_config["batch_size"], shuffle=False)

    if "binary" in dataset_name:
        data_type = "binary"
        output_size = input_size
    elif "gaussian" in dataset_name:
        data_type = "gaussian"
        output_size = 2 * input_size
    else:
        raise ValueError("Data type must be binary or Gaussian!")
    hidden_size_mults = [experiment_config[f"hidden_size_multiplier_{i}"] for i in range(1, 6)]

    wandb.init(project=args.wandb_name, config=experiment_config)

    for num_hidden_layers in range(1, 10):
        hidden_sizes = [h * input_size for h in hidden_size_mults[:num_hidden_layers]]

        # Model initialization
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
            model=model,
            optimizer=optimizer,
            train_dl=train_dl,
            val_dl=val_dl,
            max_epoch=experiment_config["max_epochs"],
            patience=experiment_config["patience"]
        )

        log_metrics_to_wandb(results, num_hidden_layers)

    wandb.finish()


def log_metrics_to_wandb(results, num_hidden_layers):
    fig, ax = plt.subplots()

    epochs = list(range(1, len(results["gradient_norms_per_epoch"]) + 1))
    ax.plot(epochs, results["gradient_norms_per_epoch"], label="Gradient Norm", color='tab:green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.tick_params(axis='y')
    ax.legend(loc='upper right')

    plt.title(f'Gradient Norms for Network with {num_hidden_layers} Hidden Layers')
    plt.tight_layout()

    wandb.log({f"Gradient Norms for {num_hidden_layers} Hidden Layers": wandb.Image(fig)})

    plt.close(fig)


if __name__ == '__main__':
    main()
