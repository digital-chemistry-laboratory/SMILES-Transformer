import os

import numpy as np
import numpy.typing  # this is neeed
from CGRtools import SMILESRead, exceptions


def save_samples(
    X: np.typing.ArrayLike,
    X_OG: np.typing.ArrayLike,
    y_true: np.typing.ArrayLike,
    subfolder: str,
    base_path: str,
    indices: np.typing.ArrayLike | list,
    y_pred=np.typing.ArrayLike,
):
    """
    Saves n samples from the dataset with their true and predicted labels. The saving folder is completely emptied first

    Args:
        X (np.array | list): List of SMILES/CGR
        X_OG (np.array |list): List of SMILES
        y_true (np.array | list): List of true labels
        subfolder (str): Subfolder where to save the samples: "best", "worst"
        base_path (str): Path to the folder where to save the samples
        indices (np.array | list): List of indices to save
        y_pred (np.array | list): List of predicted labels

    Returns:
        None
    """
    print("-----------------------------------------")
    print(f"Saving samples to {base_path}/{subfolder}")
    save_path = os.path.join(base_path, subfolder)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    X, X_OG, y_true, y_pred = (
        X[indices],
        X_OG[indices],
        y_true[indices],
        y_pred[indices],
    )
    for i in range(len(y_true)):
        try:
            parser = SMILESRead.create_parser(ignore=True)
            OG_reaction = parser(X_OG[i])
            OG_reaction.clean2d()
            m = ~OG_reaction
            if type(m) == tuple:
                m = OG_reaction

            with open(os.path.join(save_path, f"{i}_CGR_graph.svg"), "w") as f:
                f.write(m.depict())
        except exceptions.ImplementationError:
            pass

        with open(os.path.join(save_path, f"{i}_reaction.svg"), "w") as f:
            f.write(OG_reaction.depict())

        with open(os.path.join(save_path, f"{i}_info.txt"), "w") as f:
            f.writelines(
                [
                    f"Full Smiles:{X_OG[i]}"
                    f"SMILES reactant: {X_OG[i].split('>>')[0]}\n",
                    f"SMILES product: {X_OG[i].split('>>')[1]}\n",
                    f"SMILES/CGR: {X[i]}\n",
                    f"True Label: {y_true[i]}\n",
                    f"Predicted Label: {y_pred[i]}\n",
                    f"Error: {abs(y_true[i] - y_pred[i])}\n",
                ]
            )
    print("-----------------------------------------")
