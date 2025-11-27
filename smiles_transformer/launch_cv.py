import yaml
import os
from smiles_transformer.main import main
from smiles_transformer.utils.path_finder import path_finder
import numpy as np
import wandb


def launch_cv(path_to_config_folder, alternative_config=None):
    # Call the main function with these parameters
    with open(
        path_finder(
            path_to_config_folder,
            "config.yaml",
            is_file=True,
        ),
        "r",
    ) as stream:
        params = yaml.safe_load(stream)
    if alternative_config is None:
        alternative_config = {}
    for category in alternative_config:
        params[category].update(alternative_config[category])
    n_folds = params["test_settings"]["n_folds"]
    if params["test_settings"]["n_folds"] == "premade":
        n_folds = len(
            os.listdir(
                path_finder(
                    prefix=params["dataset_settings"]["dataset_folder"],
                    path_from_source=params["dataset_settings"]["dataset_name"].replace(
                        ".csv", ""
                    ),
                )
            )
        )

    assert type(n_folds) is int, f"n_folds needs to be int, not {type(n_folds)}"

    # assert (params["dataset_settings"]["n_datapoints"] is None), "n_datapoints needs to be None for a CV run."
    params["general_settings"]["tags"].append("cross-val")
    cv_results = {}

    for fold in range(1, n_folds + 1):
        params.update(
            {
                "test_settings": {
                    "test_size": str(fold) + "/" + str(n_folds),
                },
            }
        )
        result = main(
            path_to_config_folder=path_to_config_folder, alternative_config=params
        )

        for key in result:
            try:
                cv_results[key] = np.append(cv_results[key], result[key])
            except KeyError:
                cv_results[key] = np.array([result[key]])
    wandb.summary.update({"individual_run_results": cv_results})
    cv_metrics = {}
    for key in cv_results:
        cv_metrics[key + "_std"] = cv_results[key].std()
        cv_metrics[key + "_mean"] = cv_results[key].mean()

    print("-----------------------------------------")
    print("CV metrics:")
    print(cv_metrics)
    print("-----------------------------------------")
    wandb.summary.update(cv_metrics)
    wandb.log(cv_metrics, commit=True)
    wandb.finish()


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
    launch_cv(args.config_folder)
    print("done!")
