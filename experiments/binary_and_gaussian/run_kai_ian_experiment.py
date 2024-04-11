import argparse
import yaml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from binary_gaussian_mnist_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator

# Set the device for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define and parse command-line arguments
parser = argparse.ArgumentParser("Runs StrNN on a synthetic dataset.")
parser.add_argument("--experiment_name", type=str, default="multimodal")
parser.add_argument("--init_method", type=str, choices=['ian', 'kaiming'], required=True)
parser.add_argument("--num_layers", type=int, required=True)
parser.add_argument("--data_seed", type=int, default=2547)
parser.add_argument("--model_seed", type=int, default=2647)
parser.add_argument("--wandb_name", type=str, required=True)
args = parser.parse_args()


def main():
    config_path = os.path.join("/h/bukesche/StructuredNNs/experiments/binary_and_gaussian", "experiment_config.yaml")

    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)
    experiment_config = configs[args.experiment_name]

    # Load data and adjacency matrix
    dataset_name = experiment_config["dataset_name"]
    adj_mtx_name = experiment_config["adj_mtx_name"]
    train_data, val_data, adj_mtx = load_data_and_adj_mtx(dataset_name, adj_mtx_name)

    input_size = len(train_data[0])
    experiment_config["input_size"] = input_size

    # Prepare DataLoader instances
    batch_size = experiment_config["batch_size"]
    train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Compute hidden sizes based on the number of layers and multipliers in the configuration
    hidden_size_mults = [experiment_config[f"hidden_size_multiplier_{i}"] for i in range(1, 6)]
    hidden_sizes = [h * input_size for h in hidden_size_mults[:args.num_layers]]

    # Determine the output size based on the data type
    data_type = "binary"
    output_size = input_size if data_type == "binary" else 2 * input_size

    # Initialize Weights & Biases
    wandb.init(project=args.wandb_name, config=experiment_config, reinit=True,
               name=f"{args.init_method.capitalize()}_layers_{args.num_layers}")

    # Initialize the model based on the selected initialization method
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
        ian_init=(args.init_method == 'ian')
    )
    model.to(device)

    # Initialize the optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=experiment_config["learning_rate"],
        eps=experiment_config["epsilon"],
        weight_decay=experiment_config["weight_decay"]
    )

    # Run the training loop
    results = train_loop(
        model,
        optimizer,
        train_dl,
        val_dl,
        experiment_config["max_epochs"],
        experiment_config["patience"],
    )

    # Log the final validation loss
    final_val_loss = results['val_losses_per_epoch'][-1]
    print(f"Final validation loss for {args.init_method} with {args.num_layers} layers: {final_val_loss}")

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()

