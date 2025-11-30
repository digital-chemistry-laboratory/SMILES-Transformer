from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertModel,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .basemodelfactory import BaseModelFactory


class TranslationModelFactory(BaseModelFactory):
    """Factory class for creating and configuring models for translseq-2-seq tasks.

    This class extends the ModelFactory

    Methods:
        create(): Initializes the model and returns a RegressionTrainer object.
        init_model(): Initializes and returns the configured model for sequence classification.
    """

    def create(self):
        """
        Creates a Trainer with a seq-2-seq model.

        This method sets up the training arguments, tokenizer, data collator, and callbacks, and
        initializes a Trainer object.

        Returns:
            Trainer: A Trainer object configured for seq-to-seq tasks.
        """

        callbacks = []
        if self.early_stopping_patience is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience,
                    early_stopping_threshold=self.early_stopping_threshold,
                ),
            )

        data_collator = DataCollatorForSeq2Seq(self.smilestokenizer)
        training_args = Seq2SeqTrainingArguments(
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            overwrite_output_dir=True,
            auto_find_batch_size=self.auto_find_batch_size,
            eval_steps=self.eval_steps,
            output_dir=self.output_dir,
            eval_strategy="steps",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=int(self.batch_size / 2),
            num_train_epochs=self.num_train_epochs,
            max_steps=self.num_train_steps,
            fp16=self.fp16,
            predict_with_generate=True,
            report_to="wandb",
            greater_is_better=False,
            logging_steps=self.logging_steps,
            save_total_limit=self.save_total_limit,
            max_grad_norm=self.max_grad_norm,
        )
        print("Final training arguments recieved:")
        print(training_args.to_json_string())
        return Seq2SeqTrainer(
            model=self.load_encoder_decoder(),
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["eval"],
            tokenizer=self.smilestokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

    def init_model(self):
        """
        Initialize the model. This function initializes the model and returns it.

        Returns:
            Huggingface Model: The initialized model.
        """

        return self.load_encoder_decoder()

    def load_encoder_decoder(self):
        """
        Load the encoder-decoder model. This function initializes the encoder-decoder model and returns it.
        Inspired by: https://medium.com/@utk.is.here/encoder-decoder-models-in-huggingface-from-almost-scratch-c318cce098ae

        Returns:
            Huggingface EncoderDecoderModel: The initialized encoder-decoder model.
        """
        encoder_config = BertConfig(
            vocab_size=self.smilestokenizer.vocab_size,
            max_position_embeddings=(
                self.max_length + 64
            ),  # this should be some large value
            **self.model_config,
            type_vocab_size=2,  # TODO: check if this is right
            is_decoder=False,
        )

        encoder = BertModel(config=encoder_config)

        decoder_config = BertConfig(
            vocab_size=self.smilestokenizer.vocab_size,
            max_position_embeddings=(
                self.max_length + 64
            ),  # this should be some large value
            **self.model_config,
            type_vocab_size=2,  # TODO: check if this is right
            is_decoder=True,
        )

        decoder = BertForMaskedLM(config=decoder_config)

        return EncoderDecoderModel(encoder=encoder, decoder=decoder)
