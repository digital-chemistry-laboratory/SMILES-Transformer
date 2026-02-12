import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from smiles_transformer.evaluation.evaluatortemplate import EvaluatorTemplate
from smiles_transformer.utils.save_samples import save_samples
from smiles_transformer.dataset.basedatasetfactory import BaseDatasetFactory
import wandb
import os
import pandas as pd
from datasets import DatasetDict
import random


class ClassificationEvaluator(EvaluatorTemplate):
    def evaluate_implementation(
        self, test_set: BaseDatasetFactory, tokenized_dataset: DatasetDict, fold:int, average: str = "binary"
    ):
        y_pred = np.argmax(
            self.trainer.predict(
                tokenized_dataset["test"].remove_columns(["label"])
            ).predictions,
            axis=1,
        )

        y_true = self.kwargs["label_encoder"].transform(np.array(test_set["label"]))

        sorted_labels = self.kwargs["label_counter"].most_common()
        most_common_id = self.kwargs["label_encoder"].transform([sorted_labels[0][0]])[
            0
        ]
        total_count = self.kwargs["label_counter"].total()
        class_probabilities = {
            label: count / total_count for label, count in sorted_labels
        }
        baseline_predictions_prob = random.choices(
            population=list(class_probabilities.keys()),  # Class labels
            weights=list(class_probabilities.values()),  # Probabilities
            k=len(test_set),  # Number of predictions to generate
        )
        baseline_predictions_prob_ids = self.kwargs["label_encoder"].transform(
            baseline_predictions_prob
        )
        
        # Random baseline: uniform random predictions across all classes
        all_class_labels = list(class_probabilities.keys())
        baseline_predictions_random = random.choices(
            population=all_class_labels,  # Class labels
            k=len(test_set),  # Number of predictions to generate (uniform distribution)
        )
        baseline_predictions_random_ids = self.kwargs["label_encoder"].transform(
            baseline_predictions_random
        )
        
        self.save_output(test_set, y_pred, y_true, fold)
        result_metrics = {
            "recall_most_frequent_baseline": recall_score(
                y_true, most_common_id * np.ones(len(y_true)), average=average
            ),
            "precision_most_frequent_baseline": precision_score(
                y_true, most_common_id * np.ones(len(y_true)), average=average
            ),
            "f1_score_most_frequent_baseline": f1_score(
                y_true, most_common_id * np.ones(len(y_true)), average=average
            ),
            "accuracy_most_frequent_baseline": accuracy_score(
                y_true, most_common_id * np.ones(len(y_true))
            ),
            "recall_class_frequencies_baseline": recall_score(
                y_true, baseline_predictions_prob_ids, average=average
            ),
            "precision_class_frequencies_baseline": precision_score(
                y_true, baseline_predictions_prob_ids, average=average
            ),
            "f1_score_class_frequencies_baseline": f1_score(
                y_true, baseline_predictions_prob_ids, average=average
            ),
            "accuracy_class_frequencies_baseline": accuracy_score(
                y_true, baseline_predictions_prob_ids
            ),
            "recall_random_baseline": recall_score(
                y_true, baseline_predictions_random_ids, average=average
            ),
            "precision_random_baseline": precision_score(
                y_true, baseline_predictions_random_ids, average=average
            ),
            "f1_score_random_baseline": f1_score(
                y_true, baseline_predictions_random_ids, average=average
            ),
            "accuracy_random_baseline": accuracy_score(
                y_true, baseline_predictions_random_ids
            ),
            "precision": precision_score(y_true, y_pred, average=average),
            "recall": recall_score(y_true, y_pred, average=average),
            "f1_score": f1_score(y_true, y_pred, average=average),
            "accuracy": accuracy_score(y_true, y_pred),
        }
        print("result metrics:", result_metrics)
        
        # Log metrics to W&B with error handling
        try:
            self.wandb_run.summary.update(result_metrics)
            self.wandb_run.log(result_metrics, commit=True)
        except Exception as e:
            print(f"Warning: Failed to log metrics to W&B: {e}")
            print("Continuing without W&B logging. Metrics are still computed and returned.")
        
        return result_metrics

    def save_output(self, X_test, y_pred, y_true, fold):
        """
        Save the output of the model. More precisely, save the predicted and true yields, as well as the SMILES/CGR strings.

        Args:
            X_test (np.array): Array of test samples.
            y_pred (np.array): Array of predicted yields.
            y_true (np.array): Array of true yields.

        Returns:
            None
        """
        base_path = os.path.join(self.output_dir, "results/")
        os.makedirs(base_path, exist_ok=True)

        data_table = pd.DataFrame(
            {
                "smiles_CGR": X_test["text"].to_numpy(),
                "pred": y_pred.flatten(),
                "true": y_true.flatten(),
            }
        )

        data_table["original_input"] = X_test["original_input"].to_numpy()
        self.save_results_csv(data_table, base_path=base_path, fold=fold)
        
        # Log to W&B with error handling to prevent crashes on network timeouts
        try:
            samples = wandb.Table(dataframe=data_table)
            self.wandb_run.log({"Samples": samples})
        except Exception as e:
            print(f"Warning: Failed to log samples to W&B: {e}")
            print("Continuing without W&B logging. Results are still saved to results.csv")

        try:
            indices = np.arange(0, len(y_pred))
            save_samples(
                X=X_test["text"].to_numpy(),
                X_OG=X_test["original_input"].to_numpy(),
                y_true=y_true,
                y_pred=y_pred,
                indices=indices,
                subfolder="all",
                base_path=self.output_dir,
            )

        except Exception as e:
            print(f"Error while trying to save samples: {e}")
