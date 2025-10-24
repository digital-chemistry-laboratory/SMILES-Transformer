import wandb

from smiles_transformer.main import main
from smiles_transformer.utils.path_finder import path_finder
import json


def train(path_to_config_folder):
    # Call the main function with these parameters
    wandb.init()
    config = wandb.config
    main(
        alternative_config={
            "training_settings": {
                "learning_rate": config.learning_rate,
                "hidden_dropout": config.dropout,
            },
        }
    )


# TODO: fix this
with open(
    path_finder(path_to_config_folder, "config.yaml", is_file=True), "r"
) as stream:
    sweep_config = json.load(stream)


# Create the sweep
sweep_id = wandb.sweep(sweep_config, project="Smiles_CGR_transformer")

# Run the sweep, passing the configuration explicitly
wandb.agent(sweep_id, function=train)
