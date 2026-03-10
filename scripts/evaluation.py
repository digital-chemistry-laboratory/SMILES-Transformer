from scipy.stats import sem
import numpy as np
import ast
import re


# adjust paths
all_paths = {
    "e2sn2": {
        "rxn": "add/path/to/log/file",
        "SMILES/CGR": "",
        "sr-SMILES": "",
    },
}


metrics=["MAE", "MSE", "R2"]

def read_metric(metric, lines):
    results = []
    for line in lines:
        if line.startswith(f"{metric}: [["):
            read_metrics = ast.literal_eval(re.findall(r"\[.*\]", line)[0])
            results.append(np.array(read_metrics).T)
    return results

for dataset_name, paths in all_paths.items():
    print(f"\n\n\n\n\nDataset: {dataset_name}")
    for key, filepath in paths.items():
        print(f"\nFor {key}:")
        with open(filepath) as f:
            lines = f.readlines()

        for metric in metrics:
            print(f"\n{metric}:")

            for i, scores_matrix in enumerate(read_metric(metric, lines)):
                repeated_scores = scores_matrix.flatten()

                repeated_mean = np.mean(repeated_scores)
                repeated_sem = sem(repeated_scores)

                print(f"  [Match {i+1}] Mean repeated CV: {repeated_mean:.4f}")
                print(f"  [Match {i+1}] Standard error of mean repeated CV 1: {repeated_sem:.4f}")
