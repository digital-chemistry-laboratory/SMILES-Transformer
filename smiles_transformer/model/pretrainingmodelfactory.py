from transformers import (
    AutoModelForMaskedLM,
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    ElectraConfig,
    ElectraForMaskedLM,
    Trainer,
    TrainingArguments,
)

from .basemodelfactory import BaseModelFactory


class PretrainingModelFactory(BaseModelFactory):
    """
    Factory class for creating and initializing models specifically for pre-training purposes.

    This class extends ModelFactory and is designed to initialize and configure models like BERT
    or ELECTRA for pre-training tasks such as masked language modeling.

    Methods:
        create(): Creates and configures a Trainer object for pre-training.
        init_model(): Initializes and returns the specific model based on the configuration.
    """

    def create(self):
        """
        Creates and returns a Trainer object configured for pre-training.

        This method reads the model configuration, sets up training arguments, and initializes a
        Trainer object with the necessary configurations for pre-training tasks.

        Returns:
            Trainer: A Hugging Face Trainer object configured for pre-training with the specified model.
        """

        training_args = TrainingArguments(
            eval_steps=self.eval_steps,
            output_dir=self.output_dir,
            eval_strategy="steps",
            num_train_epochs=self.num_train_epochs,
            max_steps=self.num_train_steps,
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
        self.bert_config = BertConfig(
            vocab_size=self.smilestokenizer.vocab_size, **self.model_config
        )
        self.electra_config = ElectraConfig(
            vocab_size=self.smilestokenizer.vocab_size, **self.model_config
        )

        callbacks = []
        if self.early_stopping_patience is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_threshold=self.early_stopping_threshold,
                ),
            )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.smilestokenizer,
            mlm=True,
            mlm_probability=self.mask_prob,
        )

        return Trainer(
            model=None,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["eval"],
            args=training_args,
            data_collator=data_collator,
            tokenizer=self.smilestokenizer,
            callbacks=callbacks,
            model_init=self.init_model,
        )

    def init_model(self):
        """
        Initializes and returns the model for pre-training.

        Based on the configuration, this method initializes either a BERT or an ELECTRA model.
        If a pre-trained model path is provided, it initializes the model from that path;
        otherwise, it creates a new model.

        Returns:
            Union[BertForMaskedLM, ElectraForMaskedLM]: The initialized pre-training model.

        Raises:
            AssertionError: If the required configurations are not set before calling this method.
        """
        assert hasattr(self, "bert_config") and hasattr(
            self, "electra_config"
        ), self.init_before_create_error
        if self.pretrained_model_path is not None:
            return AutoModelForMaskedLM.from_pretrained(self.pretrained_model_path)
        if self.model_type == "bert":
            return BertForMaskedLM(self.bert_config)
        if self.model_type == "electra":
            return ElectraForMaskedLM(self.electra_config)
