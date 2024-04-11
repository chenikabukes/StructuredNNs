import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb

from binary_gaussian_train_utils import train_loop, load_data_and_adj_mtx
from strnn.models.strNNDensityEstimator import StrNNDensityEstimator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("Runs StrNN on synthetic dataset.")
parser.add_argument("--experiment_names", type=str, nargs='+', help="List of experiment names")
parser.add_argument("--wandb_name", type=str)
parser.add_argument("--num_trials", type=int, default=1,
                    help="Number of trials for each experiment to compute confidence intervals")
args = parser.parse_args()


def main():
    with open("./experiment_config.yaml", "r") as f:
        configs = yaml.safe_load(f)

    sparsity_levels = [0.975, 0.95, 0.875, 0.8, 0.7, 0.6, 0.5]
    run = wandb.init(project=args.wandb_name, reinit=True)

    # Initializing lists to store final validation losses for each trial
    final_val_losses_ian = {level: [] for level in sparsity_levels}
    final_val_losses_kaiming = {level: [] for level in sparsity_levels}
    # final_val_losses_ian_normal = {level: [] for level in sparsity_levels}
    # final_val_losses_kaiming_normal = {level: [] for level in sparsity_levels}

    for i, experiment_name in enumerate(args.experiment_names):
        for trial in range(args.num_trials):
            experiment_config = configs[experiment_name]
            train_data, val_data, adj_mtx = load_data_and_adj_mtx(
                experiment_config["dataset_name"], experiment_config["adj_mtx_name"])
            input_size = len(train_data[0])
            batch_size = experiment_config["batch_size"]
            num_hidden_layers = experiment_config["num_hidden_layers"]
            hidden_size_mults = [experiment_config[f"hidden_size_multiplier_{i}"] for i in range(1, 4)]
            hidden_sizes = [h * input_size for h in hidden_size_mults]
            hidden_sizes = tuple(hidden_sizes[:num_hidden_layers])
            data_type = "binary"

            # Ian Init Model Training
            model_ian = StrNNDensityEstimator(
                nin=input_size, hidden_sizes=hidden_sizes, nout=input_size,
                opt_type=experiment_config["opt_type"], opt_args={}, precomputed_masks=None,
                adjacency=adj_mtx, activation=experiment_config["activation"],
                data_type=data_type, init=1
            ).to(device)
            optimizer_ian = AdamW(model_ian.parameters(),
                                  lr=experiment_config["learning_rate"],
                                  eps=experiment_config["epsilon"],
                                  weight_decay=experiment_config["weight_decay"])
            results_ian = train_loop(model_ian,
                                     optimizer_ian,
                                     DataLoader(train_data, batch_size=batch_size, shuffle=True),
                                     DataLoader(val_data, batch_size=batch_size, shuffle=False),
                                     experiment_config["max_epochs"],
                                     experiment_config["patience"])

            final_val_losses_ian[sparsity_levels[i]].append(results_ian['val_losses_per_epoch'][-1])

            # Kaiming Init Model Training
            model_kaiming = StrNNDensityEstimator(
                nin=input_size, hidden_sizes=hidden_sizes, nout=input_size,
                opt_type=experiment_config["opt_type"], opt_args={}, precomputed_masks=None,
                adjacency=adj_mtx, activation=experiment_config["activation"],
                data_type=data_type, init=2
            ).to(device)

            optimizer_kaiming = AdamW(model_kaiming.parameters(),
                                      lr=experiment_config["learning_rate"],
                                      eps=experiment_config["epsilon"],
                                      weight_decay=experiment_config["weight_decay"])

            results_kaiming = train_loop(model_kaiming,
                                         optimizer_kaiming,
                                         DataLoader(train_data, batch_size=batch_size, shuffle=True),
                                         DataLoader(val_data, batch_size=batch_size, shuffle=False),
                                         experiment_config["max_epochs"],
                                         experiment_config["patience"])

            final_val_losses_kaiming[sparsity_levels[i]].append(results_kaiming['val_losses_per_epoch'][-1])

            # # Ian_normal Init Model Training
            # model_ian_normal = StrNNDensityEstimatorNormalisation(
            #     nin=input_size, hidden_sizes=hidden_sizes, nout=input_size,
            #     opt_type=experiment_config["opt_type"], opt_args={}, precomputed_masks=None,
            #     adjacency=adj_mtx, activation=experiment_config["activation"],
            #     data_type=data_type, init=3
            # ).to(device)
            # optimizer_ian_normal = AdamW(model_ian_normal.parameters(),
            #                       lr=experiment_config["learning_rate"],
            #                       eps=experiment_config["epsilon"],
            #                       weight_decay=experiment_config["weight_decay"])
            # results_ian_normal = train_loop(model_ian_normal,
            #                          optimizer_ian_normal,
            #                          DataLoader(train_data, batch_size=batch_size, shuffle=True),
            #                          DataLoader(val_data, batch_size=batch_size, shuffle=False),
            #                          experiment_config["max_epochs"],
            #                          experiment_config["patience"])
            #
            # final_val_losses_ian_normal[sparsity_levels[i]].append(results_ian_normal['val_losses_per_epoch'][-1])
            #
            # # Kaiming_normal Init Model Training
            # model_kaiming_normal = StrNNDensityEstimatorNormalisation(
            #     nin=input_size, hidden_sizes=hidden_sizes, nout=input_size,
            #     opt_type=experiment_config["opt_type"], opt_args={}, precomputed_masks=None,
            #     adjacency=adj_mtx, activation=experiment_config["activation"],
            #     data_type=data_type, init=4
            # ).to(device)
            #
            # optimizer_kaiming_normal = AdamW(model_kaiming_normal.parameters(),
            #                           lr=experiment_config["learning_rate"],
            #                           eps=experiment_config["epsilon"],
            #                           weight_decay=experiment_config["weight_decay"])
            #
            # results_kaiming_normal = train_loop(model_kaiming_normal,
            #                              optimizer_kaiming_normal,
            #                              DataLoader(train_data, batch_size=batch_size, shuffle=True),
            #                              DataLoader(val_data, batch_size=batch_size, shuffle=False),
            #                              experiment_config["max_epochs"],
            #                              experiment_config["patience"])
            #
            # final_val_losses_kaiming_normal[sparsity_levels[i]].append(results_kaiming_normal['val_losses_per_epoch'][-1])

    # Computing mean and standard deviation for confidence intervals
    mean_ian = np.array([np.mean(final_val_losses_ian[level]) for level in sparsity_levels])
    std_ian = np.array([np.std(final_val_losses_ian[level]) for level in sparsity_levels])
    # mean_ian_normal = np.array([np.mean(final_val_losses_ian_normal[level]) for level in sparsity_levels])
    # std_ian_normal = np.array([np.std(final_val_losses_ian_normal[level]) for level in sparsity_levels])

    mean_kaiming = np.array([np.mean(final_val_losses_kaiming[level]) for level in sparsity_levels])
    std_kaiming = np.array([np.std(final_val_losses_kaiming[level]) for level in sparsity_levels])
    # mean_kaiming_normal = np.array([np.mean(final_val_losses_kaiming_normal[level]) for level in sparsity_levels])
    # std_kaiming_normal = np.array([np.std(final_val_losses_kaiming_normal[level]) for level in sparsity_levels])

    fig, ax = plt.subplots()
    # Plotting Ian Uniform Init with markers at specific data points
    ax.plot(sparsity_levels, mean_ian, label='Ian Uniform Init', color='blue')
    ax.fill_between(sparsity_levels, mean_ian - std_ian, mean_ian + std_ian, color='blue', alpha=0.2)
    ax.plot(sparsity_levels, mean_ian, 'bo')  # Add blue circle markers

    # Plotting Kaiming Uniform Init with markers at specific data points
    ax.plot(sparsity_levels, mean_kaiming, label='Kaiming Uniform Init', color='green')
    ax.fill_between(sparsity_levels, mean_kaiming - std_kaiming, mean_kaiming + std_kaiming, color='green', alpha=0.2)
    ax.plot(sparsity_levels, mean_kaiming, 'go')  # Add green circle markers

    ax.set_title('Synthetic: Binary d100 Final VL Over Sparsity Levels')
    ax.set_xlabel('Sparsity Level')
    ax.set_ylabel('Final Validation Loss (VL)')
    ax.set_yscale('log')
    ax.legend()
    plt.show()
    wandb.log({'Final Validation Loss Over Sparsity (Ian vs Kaiming)': wandb.Image(fig)})
    fig.savefig('validation_loss_vs_sparsity.png')
    # Ian Init
    # ax.plot(sparsity_levels, mean_ian, label='Ian Init', color='blue')
    # ax.fill_between(sparsity_levels, mean_ian - std_ian, mean_ian + std_ian, color='blue', alpha=0.2)
    #
    # # Kaiming Init
    # ax.plot(sparsity_levels, mean_kaiming, label='Kaiming Init', color='green')
    # ax.fill_between(sparsity_levels, mean_kaiming - std_kaiming, mean_kaiming + std_kaiming, color='green', alpha=0.2)
    #
    # # Ian_normal Init
    # ax.plot(sparsity_levels, mean_ian_normal, label='Ian_normal Init', color='red')
    # ax.fill_between(sparsity_levels, mean_ian_normal - std_ian_normal, mean_ian_normal + std_ian_normal, color='red',
    #                 alpha=0.2)
    #
    # # Kaiming_normal Init
    # ax.plot(sparsity_levels, mean_kaiming_normal, label='Kaiming_normal Init', color='purple')
    # ax.fill_between(sparsity_levels, mean_kaiming_normal - std_kaiming_normal, mean_kaiming_normal + std_kaiming_normal,
    #                 color='purple', alpha=0.2)
    #
    # ax.set_title('d100 Synthetic Binary Final VL Over Sparsity Levels')
    # ax.set_xlabel('Sparsity Level')
    # ax.set_ylabel('Final Validation Loss (VL)')
    # ax.set_yscale('log')
    # ax.legend()
    # plt.show()
    #
    # # Log the plot in wandb
    # wandb.log(
    #     {'Final Validation Loss Over Sparsity (Ian vs Kaiming vs Ian_normal vs Kaiming_normal)': wandb.Image(fig)})
    #
    # # Save the figure
    # fig.savefig('validation_loss_vs_sparsity_all_inits.png')

if __name__ == "__main__":
    main()
