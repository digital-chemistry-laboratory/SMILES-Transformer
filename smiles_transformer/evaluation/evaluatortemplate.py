from transformers import Trainer
import os
from abc import ABC, abstractmethod


class EvaluatorTemplate(ABC):
    def __init__(
        self,
        trainer: Trainer,
        output_dir: str,
        wandb_run=None,
        vebose=True,
        **kwargs,
    ):
        assert wandb_run is not None, "Please provide a wandb run."
        self.trainer = trainer
        self.output_dir = output_dir
        self.wandb_run = wandb_run
        self.kwargs = kwargs
        self.verbose = vebose

    def evaluate(self, test_set, tokenized_dataset, fold=None):
        print(f"Evaluating model, outputting to {self.output_dir}")
        return self.evaluate_implementation(test_set, tokenized_dataset, fold)

    @abstractmethod
    def evaluate_implementation(self, test_set, tokenized_dataset):
        pass

    def check_if_args_available(self, args: list[str] | str):
        if type(args) is str:
            args = [args]
        for arg in args:
            if arg not in self.kwargs:
                raise ValueError(f"Argument {arg} not found in kwargs.")

    def save_results_csv(self, data_table, base_path, fold):
        data_table.to_csv(
            os.path.join(base_path, f"results{f'_{fold}' if fold else ''}.csv")
        )
