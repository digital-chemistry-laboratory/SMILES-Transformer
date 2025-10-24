import json

from smiles_transformer.launch_cv import launch_cv
from smiles_transformer.utils.path_finder import path_finder


def launch_loop_from_file(path_to_config_folder):
    with open(path_finder(path_to_config_folder, "arguments.txt", is_file=True)) as f:
        lines = f.readlines()
    for line in lines:
        if line == "\n":
            continue
        parameters = json.loads(line)
        launch_cv(path_to_config_folder, parameters)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the smiles_transformer model training and evaluation."
    )
    parser.add_argument(
        "config_folder",
        type=str,
        help="Path to the configuration folder (where config.yaml is located).",
    )
    # If you want to support additional optional arguments, add them here:
    # parser.add_argument("--other", type=str, help="Some other argument")

    args = parser.parse_args()
    launch_loop_from_file(args.config_folder)
    print("done!")
