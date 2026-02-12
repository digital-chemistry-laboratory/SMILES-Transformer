import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from smiles_transformer.evaluation.evaluatortemplate import EvaluatorTemplate
from smiles_transformer.utils.save_samples import save_samples
from smiles_transformer.dataset.basedatasetfactory import BaseDatasetFactory
import wandb
import os
import pandas as pd
from datasets import DatasetDict


class RegressionEvaluator(EvaluatorTemplate):
    def evaluate_implementation(
        self, test_set: BaseDatasetFactory, tokenized_dataset: DatasetDict, fold: int
    ):
        y_pred = self.trainer.predict(tokenized_dataset["test"]).predictions

        y_true = np.array(test_set["label"]).reshape(-1, 1)
        y_pred = y_pred * self.kwargs["std"] + self.kwargs["mean"]
        self.save_output(test_set, y_pred, y_true, fold)
        result_metrics = {
            "median_baseline_MAE": mean_absolute_error(
                y_true,
                self.kwargs["target_median"] * np.ones(len(y_true)),
            ),
            "median_baseline_MSE": mean_squared_error(
                y_true,
                self.kwargs["target_median"] * np.ones(len(y_true)),
            ),
            "median_baseline_R2": r2_score(
                y_true,
                self.kwargs["target_median"] * np.ones(len(y_true)),
            ),
            "MAE": mean_absolute_error(
                y_true,
                y_pred,
            ),
            "MSE": mean_squared_error(
                y_true,
                y_pred,
            ),
            "R2": r2_score(
                y_true,
                y_pred,
            ),
        }
        print("result metrics:", result_metrics)

        # Log metrics to W&B with error handling
        try:
            self.wandb_run.summary.update(result_metrics)
            self.wandb_run.log(result_metrics, commit=True)
        except Exception as e:
            print(f"Warning: Failed to log metrics to W&B: {e}")
            print(
                "Continuing without W&B logging. Metrics are still computed and returned."
            )

        return result_metrics

    def save_output(self, X_test, yield_predicted, yield_true, fold=None):
        """
        Save the output of the model. More precisely, save the predicted and true yields, as well as the SMILES/CGR strings.

        Args:
            X_test (np.array): Array of test samples.
            yield_predicted (np.array): Array of predicted yields.
            yield_true (np.array): Array of true yields.

        Returns:
            None
        """
        base_path = os.path.join(self.output_dir, f"results")
        os.makedirs(base_path, exist_ok=True)

        data_table = pd.DataFrame(
            {
                "smiles_CGR": X_test["text"].to_numpy(),
                "pred": yield_predicted.flatten(),
                "true": yield_true.flatten(),
            }
        )
        data_table["original_smiles"] = X_test["original_input"].to_numpy()
        self.save_results_csv(data_table, base_path=base_path, fold=fold)
        # Log to W&B with error handling to prevent crashes on network timeouts
        try:
            samples = wandb.Table(dataframe=data_table)
            self.wandb_run.log({"Samples": samples})
        except Exception as e:
            print(f"Warning: Failed to log samples to W&B: {e}")
            print(
                "Continuing without W&B logging. Results are still saved to results.csv"
            )
        plt.scatter(yield_true, yield_predicted)
        plt.xlabel("True label")
        plt.ylabel("Predicted label")
        plt.plot(plt.xlim(), plt.ylim(), ls="--", c=".3")
        plt.title("Predicted vs true label")
        try:
            plt.savefig(
                os.path.join(
                    base_path,
                    "plot.png",
                ),
                dpi=1250,
            )
        except RuntimeError as e:
            print(f"Error while trying to save plot: {e}")

        differences = [
            abs(a - b)
            for a, b in zip(
                yield_predicted,
                yield_true,
            )
        ]

        worst_indices = sorted(
            range(len(differences)), key=lambda i: differences[i], reverse=True
        )[:10]
        best_indices = sorted(
            range(len(differences)),
            key=lambda i: differences[i],
            reverse=False,
        )[:10]
        try:

            save_samples(
                X=X_test["text"].to_numpy(),
                X_OG=X_test["original_input"].to_numpy(),
                y_true=yield_true,
                y_pred=yield_predicted,
                indices=best_indices,
                subfolder="best",
                base_path=self.output_dir,
            )
            save_samples(
                X=X_test["text"].to_numpy(),
                X_OG=X_test["original_input"].to_numpy(),
                y_true=yield_true,
                y_pred=yield_predicted,
                indices=worst_indices,
                subfolder="worst",
                base_path=self.output_dir,
            )
        except Exception as e:
            print(f"Error while trying to save samples: {e}")
