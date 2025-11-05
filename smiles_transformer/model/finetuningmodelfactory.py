from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)
from torch import nn
from smiles_transformer.model.featureaugmentedclassificationbert import (
    FeatureAugmentedClassificationBERT,
)
from smiles_transformer.modelization.compute_metric_for_classification import (
    compute_metrics_for_classification,
)
from smiles_transformer.modelization.compute_metric_for_regression import (
    compute_metrics_for_regression,
)


from .basemodelfactory import BaseModelFactory


class FinetuningModelFactory(BaseModelFactory):
    """Factory class for creating and configuring models for fine-tuning tasks.

    This class extends the ModelFactory for the specific purpose of fine-tuning pre-trained models,
    particularly for sequence classification tasks.

    Methods:
        create(): Initializes the model and returns a RegressionTrainer object.
        init_model(): Initializes and returns the configured model for sequence classification.
    """

    no_pretrain_path_error = "No pretrained model path was specified."

    def create(self):
        """
        Creates a RegressionTrainer with a fine-tuned model.

        This method sets up the training arguments, tokenizer, data collator, and callbacks, and
        initializes a RegressionTrainer object for fine-tuning a pre-trained model on a specific dataset.

        Raises:
            AssertionError: If no pretrained model path is specified.

        Returns:
            RegressionTrainer: A RegressionTrainer object configured for fine-tuning.
        """

        assert self.pretrained_model_path is not None, self.no_pretrain_path_error
        """
        Load the model. This function initializes the model and returns a trainer object. This methods needs to be called BEFORE the reinit method.
        """

        training_args = TrainingArguments(
            eval_steps=self.eval_steps,
            output_dir=self.output_dir,
            eval_strategy="steps",
            num_train_epochs=self.num_train_epochs
            or 1,  # leave this, the new version of transformers requires that this not be a None.
            max_steps=self.num_train_steps,  # Overrides num_train_epochs
            report_to="wandb",
            learning_rate=self.learning_rate,
            auto_find_batch_size=self.auto_find_batch_size,
            per_device_train_batch_size=self.batch_size,
            overwrite_output_dir=True,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            fp16=self.fp16,
            warmup_ratio=self.warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=self.logging_steps,
            save_total_limit=self.save_total_limit,
            seed=self.random_state if self.random_state is not None else 42,
        )

        callbacks = []
        if self.early_stopping_patience is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_threshold=self.early_stopping_threshold,
                ),
            )

        data_collator = DataCollatorWithPadding(tokenizer=self.smilestokenizer)
        if self.n_labels > 1:
            compute_metrics = compute_metrics_for_classification
        else:
            compute_metrics = compute_metrics_for_regression
        
        return Trainer(
            model=self.init_model(),
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["eval"],
            args=training_args,
            tokenizer=self.smilestokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            model_init=self.init_model,
        )

    def init_model(self):
        """
        Initialize the model. This function initializes the model and returns it.

        Returns:
            Huggingface Model: The initialized model.
        """
        if self.additional_features:
            return FeatureAugmentedClassificationBERT.from_pretrained_downstream(
                self.pretrained_model_path,
                num_labels=self.n_labels,
                n_features=len(self.additional_features),
                layer_width=self.layer_width,
                n_layers=self.n_layers,
            )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model_path,
            num_labels=self.n_labels,
        )
        layers = [
            nn.Linear(
                in_features=model.config.hidden_size,
                out_features=self.layer_width,
            ),
            nn.ReLU(),
            nn.Dropout(self.hidden_dropout),
        ]
        for i in range(self.n_layers):
            layers.append(
                nn.Linear(in_features=self.layer_width, out_features=self.layer_width)
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hidden_dropout))
        layers.append(
            nn.Linear(in_features=self.layer_width, out_features=self.n_labels)
        )
        if self.n_labels > 1:
            layers.append(nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(*layers)

        return model
