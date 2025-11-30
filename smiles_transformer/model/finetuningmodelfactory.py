from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    TrainerCallback,
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


class FreezeUnfreezeCallback(TrainerCallback):
    """
    Callback to freeze encoder layers at the start of training and unfreeze them after a specified number of steps.

    Args:
        freeze_steps: Number of steps to keep encoder frozen
        gradual: If True, gradually unfreezes layers one at a time
        model: The model instance
    """

    def __init__(self, freeze_steps, gradual=False, model=None):
        self.freeze_steps = freeze_steps
        self.gradual = gradual
        self.model = model
        self.encoder_frozen = False
        self.num_encoder_layers = None

    def freeze_encoder(self, model):
        """Freeze all encoder layers except the classification head."""
        # Freeze embeddings
        if hasattr(model, 'bert'):
            base_model = model.bert
        elif hasattr(model, 'roberta'):
            base_model = model.roberta
        elif hasattr(model, 'electra'):
            base_model = model.electra
        else:
            # Try to find encoder attribute
            base_model = getattr(model, 'encoder', None)
            if base_model is None:
                print("Warning: Could not find encoder to freeze")
                return

        # Freeze embeddings
        if hasattr(base_model, 'embeddings'):
            for param in base_model.embeddings.parameters():
                param.requires_grad = False

        # Freeze encoder layers
        if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
            self.num_encoder_layers = len(base_model.encoder.layer)
            for layer in base_model.encoder.layer:
                for param in layer.parameters():
                    param.requires_grad = False

        self.encoder_frozen = True
        print(f"Encoder frozen for the first {self.freeze_steps} steps")

    def unfreeze_encoder(self, model, step):
        """Unfreeze encoder layers (all at once or gradually)."""
        if hasattr(model, 'bert'):
            base_model = model.bert
        elif hasattr(model, 'roberta'):
            base_model = model.roberta
        elif hasattr(model, 'electra'):
            base_model = model.electra
        else:
            base_model = getattr(model, 'encoder', None)
            if base_model is None:
                return

        if self.gradual and self.num_encoder_layers:
            # Gradual unfreezing: unfreeze from top layer (closest to output) to bottom
            steps_per_layer = self.freeze_steps / self.num_encoder_layers
            layers_to_unfreeze = int((step / steps_per_layer))

            if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
                for i in range(self.num_encoder_layers - 1, self.num_encoder_layers - 1 - layers_to_unfreeze, -1):
                    if i >= 0:
                        for param in base_model.encoder.layer[i].parameters():
                            param.requires_grad = True
        else:
            # Unfreeze all at once
            if hasattr(base_model, 'embeddings'):
                for param in base_model.embeddings.parameters():
                    param.requires_grad = True

            if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'):
                for layer in base_model.encoder.layer:
                    for param in layer.parameters():
                        param.requires_grad = True

            print(f"Encoder unfrozen at step {step}")
            self.encoder_frozen = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Freeze encoder at the start of training."""
        if self.freeze_steps and self.freeze_steps > 0:
            self.model = model
            self.freeze_encoder(model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Check if it's time to unfreeze the encoder."""
        if self.freeze_steps and state.global_step >= self.freeze_steps and self.encoder_frozen:
            self.unfreeze_encoder(model, state.global_step)
        elif self.freeze_steps and self.gradual and state.global_step < self.freeze_steps:
            # For gradual unfreezing, unfreeze progressively
            self.unfreeze_encoder(model, state.global_step)


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
            max_grad_norm=self.max_grad_norm,
            weight_decay=self.weight_decay,
            seed=self.random_state if self.random_state is not None else 42,
            label_smoothing_factor=self.label_smoothing,
        )

        callbacks = []
        if self.early_stopping_patience is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_threshold=self.early_stopping_threshold,
                ),
            )

        # Add freeze/unfreeze callback if configured
        if self.freeze_encoder_steps is not None and self.freeze_encoder_steps > 0:
            callbacks.append(
                FreezeUnfreezeCallback(
                    freeze_steps=self.freeze_encoder_steps,
                    gradual=self.gradual_unfreezing,
                )
            )

        data_collator = DataCollatorWithPadding(tokenizer=self.smilestokenizer)
        if self.n_labels > 1:
            compute_metrics = compute_metrics_for_classification
        else:
            compute_metrics = compute_metrics_for_regression
        print(f"Final length of training dataset: {len(self.dataset['train'])}")
        if "eval" in self.dataset:
            print(f"Final length of eval dataset: {len(self.dataset['eval'])}")
        if "test" in self.dataset:
            print(f"Final length of test dataset: {len(self.dataset['test'])}")
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
            model = FeatureAugmentedClassificationBERT.from_pretrained_downstream(
                self.pretrained_model_path,
                num_labels=self.n_labels,
                n_features=len(self.additional_features),
                layer_width=self.layer_width,
                n_layers=self.n_layers,
            )
        else:
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

        # Apply LayerDrop if configured
        if self.layerdrop_prob is not None and self.layerdrop_prob > 0:
            # Check if model has the config attribute for layerdrop
            if hasattr(model.config, 'layerdrop'):
                model.config.layerdrop = self.layerdrop_prob
                print(f"LayerDrop enabled with probability {self.layerdrop_prob}")
            else:
                print(f"Warning: LayerDrop requested but not supported by {self.model_type}")

        # Apply Stochastic Depth if configured
        if self.stochastic_depth_prob is not None and self.stochastic_depth_prob > 0:
            # This would typically require modifying the model architecture
            # For now, we'll add a warning that this needs custom implementation
            print(f"Warning: Stochastic Depth with prob {self.stochastic_depth_prob} requested but requires custom model implementation")

        return model
