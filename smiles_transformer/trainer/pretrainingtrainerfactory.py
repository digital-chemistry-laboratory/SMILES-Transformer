import pandas as pd

from smiles_transformer.utils.optuna_hp_space import optuna_hp_space
from smiles_transformer.utils.path_finder import path_finder

from .basetrainerfactory import BaseTrainerFactory


class PretrainingTrainerFactory(BaseTrainerFactory):
    """
    A factory class for creating trainers specialized in pretraining models,
    inheriting from the TrainerFactory class.

    This class focuses on setting up and initializing training configurations
    and environments for pretraining tasks, specifically for models dealing
    with SMILES (Simplified Molecular Input Line Entry System) data.
    """

    def create_and_train(self, dataset: pd.DataFrame):
        """
        Creates and configures a pretraining environment and trainer using the provided data. Trains the model and returns the trainer and wandb run.

        The method initializes a training environment, prepares the dataset,
        and creates a model trainer for pretraining tasks. It handles model
        checkpoints, training configurations, and integrates with Weights & Biases
        (wandb) for experiment tracking.

        Args:
            dataset (pd.DataFrame): A DataFrame containing the dataset to be used for pretraining.
                                 The dataset must include a column with SMILES strings.

        Returns:
            A tuple containing the configured trainer.
            If hyperparameter optimization is enabled, it returns the best trials.

        Notes:
            - The method assumes the presence of specific attributes from the parent class,
              such as `self.smiles_column_name`, `self.test_mode`, `self.model_type`, etc.
            - It handles renaming of columns, splitting the dataset, setting up the dataset
              and model factories, and initializing Weights & Biases (wandb) runs.
            - The method manages different scenarios such as resuming from the last checkpoint,
              running hyperparameter optimization, and saving the trained model.
        """

        dataset.rename(columns={self.features_column_name: "text"}, inplace=True)
        self.dataset = dataset
        self.output_dir = path_finder(
            prefix=self.path_to_outputs,
            path_from_source=f"/pretraining/{self.model_type}/{self.model_size}/{self.tokenizer_kind}/{self.run.name}",
            is_file=False,
            create_path_if_unavailable=True,
        )
        self.test_size = 0
        X_train, X_eval = self.train_test_split(
            self.dataset,
            splits=["train", "eval"],
            split_sizes={"test": self.test_size, "eval": self.eval_size},
        )
        self.check_eval_length(X_eval)

        self.tokenized_dataset = self.dataset_factory.create(
            X_train=X_train,
            y_train=None,
            X_eval=X_eval,
        )

        if (self.run_name is not None) and (self.checkpoint_number is not None):
            self.output_dir = self.pretrained_model_path

        model_factory = self.model_factory(
            output_dir=self.output_dir,
            model_type=self.model_type,
            dataset=self.tokenized_dataset,
            eval_steps=self.eval_steps,
            num_train_epochs=self.num_train_epochs,
            smilestokenizer=self.smilestokenizer,
            num_train_steps=self.num_train_steps,
            pretrained_model_path=self.pretrained_model_path,
            early_stopping_patience=self.early_stopping_patience,
            early_stopping_threshold=self.early_stopping_threshold,
            learning_rate=self.learning_rate,
            mask_prob=self.mask_prob,
            max_length=self.max_length,
            model_size=self.model_size,
            hidden_dropout=self.hidden_dropout,
            warmup_ratio=self.warmup_ratio,
            additional_features=self.additional_features,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            batch_size=self.batch_size,
            auto_find_batch_size=self.auto_find_batch_size,
            fp16=self.fp16,
            logging_steps=self.logging_steps,
            save_total_limit=self.save_total_limit,
            max_grad_norm=self.max_grad_norm,
            weight_decay=self.weight_decay,
            random_state=self.random_state,
        )

        trainer = model_factory.create()

        if self.n_trials is not None:
            best_trials = trainer.hyperparameter_search(
                direction="minimize",
                backend="optuna",
                hp_space=optuna_hp_space,
                n_trials=self.n_trials,
            )
            print("\n", best_trials, "\n")
            return best_trials, self.run
            
        trainer.train(resume_from_checkpoint=self.get_checkpoint_folder())
        trainer.save_model(self.output_dir)
        self.trainer = trainer
        self.latest_run_handler(self.run.name, read=False)

        return trainer

    def define_tokenizer(self, smilestokenizer):
        self.smilestokenizer = smilestokenizer(self.path_to_vocab_file)
        self.tokenizer_kind = self.smilestokenizer.tokenizer_kind
        self.pretrained_model_path = None

        if self.run_name == "last":
            self.run_name = self.latest_run_handler(
                self.run.name, read=(self.run_name == "last")
            )
        if self.run_name is not None:
            self.pretrained_model_path = path_finder(
                prefix=self.path_to_outputs,
                path_from_source=f"pretraining/{self.model_type}/{self.model_size}/{self.tokenizer_kind}/{self.run_name}",
                is_file=False,
            )
            self.smilestokenizer = smilestokenizer.from_pretrained(
                self.pretrained_model_path
            )
