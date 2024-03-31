import argparse
import yaml
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from binary_gaussian_mnist_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimatorNormalisation import StrNNDensityEstimatorNormalisation

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
    model = StrNNDensityEstimatorNormalisation(
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

    # After the loop, plot the final validation losses for both initialization methods
    # fig, ax = plt.subplots()
    # ax.plot(range(1, 4), final_val_losses_ian, marker='o', linestyle='-', label='Ian Init')
    # ax.plot(range(1, 4), final_val_losses_kaiming, marker='o', linestyle='-', label='Kaiming Init')
    # ax.set_title('Final Validation Loss Over Layers (d100)')
    # ax.set_xlabel('Number of Layers')
    # ax.set_ylabel('Final Validation Loss')
    # ax.legend()
    # wandb.log({'Final Validation Loss Over Layers (Ian vs Kaiming)': wandb.Image(fig)})
    # plt.close(fig)
    #
    # wandb.finish()

#
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
#     data_type = "binary"
#     output_size = input_size if data_type == "binary" else 2 * input_size
#
#     run = wandb.init(project=args.wandb_name, config=experiment_config, reinit=True)
#
#     num_runs = 10
#     final_val_losses_ian = defaultdict(list)
#     final_val_losses_kaiming = defaultdict(list)
#
#     for num_layers in range(1, 10):
#         hidden_sizes = [h * input_size for h in hidden_size_mults[:num_layers]]
#
#         for run_idx in range(num_runs):
#             model_ian = StrNNDensityEstimatorNormalisation(
#                 nin=input_size,
#                 hidden_sizes=hidden_sizes,
#                 nout=output_size,
#                 opt_type=experiment_config["opt_type"],
#                 opt_args={},
#                 precomputed_masks=None,
#                 adjacency=adj_mtx,
#                 activation=experiment_config["activation"],
#                 data_type=data_type,
#                 ian_init=True
#             )
#             model_ian.to(device)
#
#             optimizer_ian = AdamW(
#                 model_ian.parameters(),
#                 lr=experiment_config["learning_rate"],
#                 eps=experiment_config["epsilon"],
#                 weight_decay=experiment_config["weight_decay"]
#             )
#
#             results_ian = train_loop(
#                 model_ian,
#                 optimizer_ian,
#                 train_dl,
#                 val_dl,
#                 experiment_config["max_epochs"],
#                 experiment_config["patience"],
#             )
#             final_val_losses_ian[num_layers].append(results_ian['val_losses_per_epoch'][-1])
#
#             model_kaiming = StrNNDensityEstimatorNormalisation(
#                 nin=input_size,
#                 hidden_sizes=hidden_sizes,
#                 nout=output_size,
#                 opt_type=experiment_config["opt_type"],
#                 opt_args={},
#                 precomputed_masks=None,
#                 adjacency=adj_mtx,
#                 activation=experiment_config["activation"],
#                 data_type=data_type,
#                 ian_init=False
#             )
#             model_kaiming.to(device)
#
#             optimizer_kaiming = AdamW(
#                 model_kaiming.parameters(),
#                 lr=experiment_config["learning_rate"],
#                 eps=experiment_config["epsilon"],
#                 weight_decay=experiment_config["weight_decay"]
#             )
#
#             results_kaiming = train_loop(
#                 model_kaiming,
#                 optimizer_kaiming,
#                 train_dl,
#                 val_dl,
#                 experiment_config["max_epochs"],
#                 experiment_config["patience"]
#             )
#             final_val_losses_kaiming[num_layers].append(results_kaiming['val_losses_per_epoch'][-1])
#
#     # Convert dictionaries to lists for plotting
#     final_val_losses_ian_list = [final_val_losses_ian[num_layers] for num_layers in range(1, 10)]
#     final_val_losses_kaiming_list = [final_val_losses_kaiming[num_layers] for num_layers in range(1, 10)]
#
#     # Calculate mean and confidence interval for Ian Init
#     mean_ian = np.mean(final_val_losses_ian_list, axis=1)
#     ci_ian = stats.sem(final_val_losses_ian_list, axis=1) * stats.t.ppf((1 + 0.95) / 9,
#                                                                         len(final_val_losses_ian_list[0]) - 1)
#
#     # Calculate mean and confidence interval for Kaiming Init
#     mean_kaiming = np.mean(final_val_losses_kaiming_list, axis=1)
#     ci_kaiming = stats.sem(final_val_losses_kaiming_list, axis=1) * stats.t.ppf((1 + 0.95) / 9, len(
#         final_val_losses_kaiming_list[0]) - 1)
#
#     # Plotting with confidence intervals as bands
#     fig, ax = plt.subplots()
#
#     # Plot the mean line for Ian Init and Kaiming Init
#     ax.plot(range(1, 10), mean_ian, marker='o', linestyle='-', label='Ian Init', color='blue')
#     ax.plot(range(1, 10), mean_kaiming, marker='o', linestyle='-', label='Kaiming Init', color='green')
#
#     # Create the confidence bands for Ian Init
#     ax.fill_between(range(1, 10), mean_ian - ci_ian, mean_ian + ci_ian, color='blue', alpha=0.2)
#
#     # Create the confidence bands for Kaiming Init
#     ax.fill_between(range(1, 10), mean_kaiming - ci_kaiming, mean_kaiming + ci_kaiming, color='green', alpha=0.2)
#
#     ax.set_title('Final Validation Loss Over Layers (d100) with 95% CI')
#     ax.set_xlabel('Number of Layers')
#     ax.set_ylabel('Final Validation Loss')
#     ax.legend()
#     wandb.log({'Final Validation Loss Over Layers (Ian vs Kaiming) with CI': wandb.Image(fig)})
#     plt.close(fig)
#
#     wandb.finish()
