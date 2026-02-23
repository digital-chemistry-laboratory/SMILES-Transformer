from collections import defaultdict
import json
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
        fold_root = path_finder(
            prefix=params["dataset_settings"]["dataset_folder"],
            path_from_source=params["dataset_settings"]["dataset_name"].replace(".csv", ""),
        )

        n_folds = len(
            [
                f
                for f in os.listdir(fold_root)
                if not f.startswith(".") and os.path.isdir(os.path.join(fold_root, f)) # ignore .DS_Store or any (hidden) files
            ]
        )
        
    assert type(n_folds) is int, f"n_folds needs to be int, not {type(n_folds)}"

    # assert (params["dataset_settings"]["n_datapoints"] is None), "n_datapoints needs to be None for a CV run."
    params["general_settings"]["tags"].append("cross-val")
    cv_results = {}
    seeds = params["general_settings"]["random_state"]
    n_runs = params["general_settings"]["random_state"]
        
    cv_results = defaultdict(list)

    for fold_idx in range(1, n_folds + 1):
        for seed in range(seeds):
            print(f"\nStarting run with seed={seed} for fold {fold_idx} ...")
            # Update parameters for this fold/run combination
            # Use .update() on nested dicts to preserve existing values
            params["test_settings"].update({"test_size": f"{fold_idx}/{n_folds}"})
            params["general_settings"].update({"random_state": seed})

            result = main(
                path_to_config_folder=path_to_config_folder,
                alternative_config=params,
            )

            for key, value in result.items():
                if len(cv_results[key]) < fold_idx:
                    cv_results[key].append([])
                cv_results[key][fold_idx - 1].append(value)
                
    aggregated_results = {}    

    for key, fold_values in cv_results.items():
        print(f"{key}: {fold_values}")
        fold_means = [np.mean(runs) for runs in fold_values]
        fold_medians = [np.median(runs) for runs in fold_values]
        fold_stds = [np.std(runs, ddof=1) for runs in fold_values]

        mean_over_folds = np.mean(fold_medians)
        std_over_folds = np.std(fold_medians, ddof=1)

        # optional total uncertainty (fold + run stochasticity)
        total_var = std_over_folds**2 + np.mean(np.square(fold_stds))
        total_std = np.sqrt(total_var)

        aggregated_results[key] = {
            "fold_means": fold_means,
            "fold_medians": fold_medians,
            "fold_stds": fold_stds,
            "mean_over_folds": mean_over_folds,
            "std_over_folds": std_over_folds,
            "total_std": total_std,
        }

    for key, vals in aggregated_results.items():
        print(
            f"{key}: {vals['mean_over_folds']:.4f} ± {vals['std_over_folds']:.4f} "
            f"(over {n_folds} folds, {n_runs} runs per fold)"
        )

    wandb.summary.update({"individual_run_results": cv_results})
    cv_metrics = {}

    for key, fold_values in cv_results.items():
        # Convert list-of-lists to 2D NumPy array
        arr = np.array(fold_values, dtype=float)  # shape: (n_folds, n_runs)

        # Mean over runs (axis=1), then mean over folds
        fold_means = np.mean(arr, axis=1)
        mean_over_folds = np.mean(fold_means)
        std_over_folds = np.std(fold_means, ddof=1)

        cv_metrics[key + "_mean"] = mean_over_folds
        cv_metrics[key + "_std"] = std_over_folds


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
    parser.add_argument(
        "--override",
        type=str,
        help='JSON string to override config. Example: \'{"training_settings": {"learning_rate": 1e-5}}\''
    )

    args = parser.parse_args()
    
    alternative_config = json.loads(args.override) if args.override else {}
    
    launch_cv(args.config_folder, alternative_config=alternative_config)
    print("done!")
