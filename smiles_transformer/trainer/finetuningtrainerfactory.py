import pandas as pd

from smiles_transformer.model.finetuningmodelfactory import FinetuningModelFactory
from smiles_transformer.utils.optuna_hp_space import optuna_hp_space

from .basetrainerfactory import BaseTrainerFactory
from smiles_transformer.utils.path_finder import path_finder


class FinetuningTrainerFactory(BaseTrainerFactory):
    """
    A factory class for creating trainers specialized in finetuning models,
    inheriting from the TrainerFactory class.

    This class focuses on setting up and initializing training configurations
    and environments for finetuning tasks, specifically for models dealing
    with SMILES (Simplified Molecular Input Line Entry System) data.
    """

    def create_and_train(self, dataset: pd.DataFrame):
        """
        Creates and configures a finetuning environment and trainer using the provided data. Trains the model and returns the trainer.

        The method initializes a training environment, prepares the dataset,
        and creates a model trainer for finetuning tasks. It handles model
        checkpoints, training configurations, and integrates with Weights & Biases
        (wandb) for experiment tracking.

        Args:
            dataset (pd.DataFrame): A DataFrame containing the dataset to be used for finetuning.
                                 The dataset must include a column with SMILES strings.
            

        Returns:
            A tuple containing the configured trainer.
            If hyperparameter optimization is enabled, it returns the best trials.
        """
        
        assert (
            self.run_name is not None
        ), "Please provide the run name of a pretrained model. If you want to train a model from scratch, please pretrain the model first."
        assert self.label_column_name is not None, self.no_label_column_error

        dataset.rename(
            columns={
                self.label_column_name: "label",
                self.features_column_name: "text",
            },
            inplace=True,
        )
        dataset = dataset

        X_train, X_test, X_eval, y_train, y_test, y_eval = self.train_test_split(
            dataset.drop(columns=["label"]),
            dataset["label"],
            splits=["train", "test", "eval"],
            split_sizes={"test": self.test_size, "eval": self.eval_size},
        )
        self.check_eval_length(X_eval)

        self.test_set = pd.concat([X_test, y_test], axis=1)

        self.tokenized_dataset = self.dataset_factory.create(
            X_train=X_train,
            X_eval=X_eval,
            y_train=y_train,
            y_eval=y_eval,
            X_test=X_test,
            y_test=y_test,
            scale_target=self.scale_target,
        )

        model_factory = FinetuningModelFactory(
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
            lr_scheduler_type=self.lr_scheduler_type,
            additional_features=self.additional_features,
            n_layers=self.n_layers,
            n_labels=self.n_labels,
            layer_width=self.layer_width,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            batch_size=self.batch_size,
            auto_find_batch_size=self.auto_find_batch_size,
            fp16=self.fp16,
            logging_steps=self.logging_steps,
            save_total_limit=self.save_total_limit,
            max_grad_norm=self.max_grad_norm,
            weight_decay=self.weight_decay,
            random_state=self.random_state,
            label_smoothing=self.label_smoothing,
            freeze_encoder_steps=self.freeze_encoder_steps,
            gradual_unfreezing=self.gradual_unfreezing,
            layerdrop_prob=self.layerdrop_prob,
            stochastic_depth_prob=self.stochastic_depth_prob,
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
            return best_trials
        if self.skip_training:
            return trainer
        trainer.train(resume_from_checkpoint=self.get_checkpoint_folder())
        trainer.save_model(self.output_dir)
        return trainer

    def define_tokenizer(self, smilestokenizer):
        self.smilestokenizer = smilestokenizer(self.path_to_vocab_file)
        self.tokenizer_kind = self.smilestokenizer.tokenizer_kind
        self.pretrained_model_path = None
        if self.run_name == "last":
            self.run_name = self.latest_run_handler(
                self.run.name, read=(self.run_name == "last")
            )

        self.pretrained_model_path = path_finder(
            prefix=self.path_to_outputs,
            path_from_source=f'/{"finetuning" if self.skip_training else "pretraining"}/{self.model_type}/{self.model_size}/{self.tokenizer_kind}/{self.run_name}',
            is_file=False,
        )

        # replace the tokenizer:
        self.smilestokenizer = smilestokenizer.from_pretrained(
            self.pretrained_model_path
        )
