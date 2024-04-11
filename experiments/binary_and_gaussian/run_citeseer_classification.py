import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import wandb

from strnn.models.strNNDensityEstimator import StrNNDensityEstimator
from ..strnn.models.strNNClassifier import StrNNClassifier
from citeseer_classification_train_utils import load_data
from citeseer_classification_train_utils import train_model, evaluate_model

# Setup device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Argument parsing
parser = argparse.ArgumentParser(description="Runs StrNN on the CiteSeer dataset.")
parser.add_argument("--experiment_name", type=str, default="citeseer_classification")
parser.add_argument("--wandb_name", type=str, help="Weights & Biases project name")
args = parser.parse_args()


def main():
    with open("./experiment_config.yaml", "r") as f:
        configs = yaml.safe_load(f)
        experiment_config = configs[args.experiment_name]

    wandb.init(project=args.wandb_name, name=args.experiment_name, config=experiment_config)

    data, adj_matrix, num_features, num_classes = load_data()
    model_params = experiment_config['model_params']
    hidden_sizes = [model_params[f"hidden_size_multiplier_{i}"] * num_features for i in
                    range(1, model_params["num_hidden_layers"] + 1)]
    model = StrNNClassifier(num_features, num_classes, hidden_sizes, adj_matrix, init_type=1,
                            activation=model_params["activation"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=experiment_config['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(experiment_config['epochs']):
        train_loss = train_model(model, data, optimizer, criterion)
        val_loss, val_accuracy = evaluate_model(model, data, criterion, data.val_mask)

        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        print(f'Epoch {epoch}: Train Loss {train_loss}, Val Loss {val_loss}, Val Accuracy {val_accuracy}')

    model.load_state_dict(best_model_state)
    test_loss, test_accuracy = evaluate_model(model, data, criterion, data.test_mask)

    print(f'Test Loss: {test_loss}, Test Accuracy {test_accuracy}')
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})


if __name__ == '__main__':
    main()
