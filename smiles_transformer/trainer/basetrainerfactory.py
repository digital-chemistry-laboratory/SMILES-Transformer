import json
import os
import re
from abc import ABC, abstractmethod
from datetime import timedelta
from timeit import default_timer as timer

import pandas as pd

import wandb
from smiles_transformer.dataset.basedatasetfactory import BaseDatasetFactory
from smiles_transformer.model.basemodelfactory import BaseModelFactory
from smiles_transformer.tokenizer.basetokenizertemplate import BaseTokenizerTemplate
from smiles_transformer.utils.path_finder import path_finder
from smiles_transformer.splitter.basesplitterfactory import BaseSplitterFactory
import warnings


class BaseTrainerFactory(ABC):
    """
    An abstract base class for creating and configuring a Hugging Face Trainer.

    This class serves as a template for trainer factories, providing common functionality
    and enforcing the implementation of specific methods in child classes.

    Attributes:
        model_factory (BaseModelFactory): Factory for creating models.
        dataset_folder (str): path to the dataset folder.
        path_to_vocab_file (str): Path to the vocabulary file for the tokenizer.
        dataset_factory (BaseDatasetFactory): Factory for creating datasets.
        features_column_name (str): Name of the column containing SMILES strings.
        label_column_name (str, optional): Name of the column containing labels. Defaults to None.
        dataset_name (str): Name of the dataset.
        model_type (str): Type of the model to be used (e.g., 'bert').
        run_name (str): Name of the run. Defaults to None.
        num_train_epochs (int): Number of training epochs.
        num_train_steps (int): Number of training steps. Defaults to -1.
        mask_prob (float): Probability of masking for masked language modeling.
        learning_rate (float): Learning rate for the model.
        test_size (float): Proportion of dataset to include in test split.
        eval_size (float): Proportion of dataset to include in eval split.
        test_mode (bool): Whether to run in test mode.
        eval_steps (int): Number of steps for evaluation.
        model_size (str): Size of the model (e.g., 'small').
        early_stopping_patience (int, optional): Patience for early stopping. Defaults to None.
        early_stopping_threshold (float): Threshold for early stopping.
        verbose (bool): Whether to output verbose messages.
        hidden_dropout (float): Dropout rate for hidden layers.
        scale_target (bool): Whether to scale the target variable.
        gradient_accumulation_steps (int): Number of gradient accumulation steps.
        max_length (int): Maximum sequence length.
        batch_size (int): Batch size for training.
        auto_find_batch_size (bool): Whether to automatically find the optimal batch size.
        fp16 (bool): Whether to use 16-bit (mixed) precision training.
        warmup_ratio (float): Warmup ratio for learning rate scheduler.
        additional_features (list): List of additional features to include.
        n_layers (int): Number of layers in the model.
        n_labels (int): Number of labels in the dataset.
        layer_width (int): Width of each layer in the model.
        logging_steps (int): Number of steps between logging.
        checkpoint_number (int or str, optional): Specific checkpoint number to load. Defaults to None.
        save_total_limit (int, optional): Maximum number of checkpoints to save. Defaults to None.
        max_grad_norm (float): Maximum gradient norm for gradient clipping. Defaults to 1.0.
        weight_decay (float): Weight decay coefficient for L2 regularization. Defaults to 0.0.
        n_trials (int, optional): Number of trials for hyperparameter search. Defaults to None.
        random_state (int, optional): Random state for reproducibility. Defaults to None.
        save_dataset_path (str): Path to save the preprocessed dataset.
        load_dataset_path (str): Path to load the preprocessed dataset.
        params (dict): The entire configuration dictionary to be pushed to wandb.
        smilestokenizer (BaseTokenizerTemplate): Tokenizer for SMILES strings.
        train_test_split (Splitter): Splitter for splitting the dataset.
        output_dir (str): Directory to save outputs.
        run (wandb.Run): Weights and Biases run object.
        start (float): Start time for training.
        pretrained_model_path (str): Path to the pretrained model. (Should be defined in child classes if used)
        tokenizer_kind (str): Kind of tokenizer being used. (Should be defined in `define_tokenizer`)
        skip_training (bool, optional): If True, skips training and returns the model.

    Methods:
        create(dataset: pd.DataFrame):
            Entry point for creating and configuring a Hugging Face Trainer.

        create_and_train(dataset: pd.DataFrame):
            Abstract method to create and configure a Hugging Face Trainer. Must be implemented in child classes.

        last_checkpoint() -> int or None:
            Get the number of the last checkpoint.

        latest_run_handler(run_name: str, read=False):
            Handle the latest run and related information.

        get_checkpoint_folder() -> str or None:
            Get the path to the checkpoint folder.

        train_start_message():
            Display the start message for training.

        train_end_message():
            Display the end message for training.

        define_tokenizer(smilestokenizer):
            Abstract method to define the tokenizer. Must be implemented in child classes.
    """

    no_label_column_error = "model mode is set to fine tuning. Please provide a the name of the label column in the config file."
    tokenizer_not_correct_error = (
        "The tokenizer must be a subclass of BaseTokenizerTemplate"
    )

    def __init__(
        self,
        dataset_factory: BaseDatasetFactory,
        path_to_outputs: str,
        dataset_folder: str,
        model_factory: BaseModelFactory,
        dataset_name: str,
        features_column_name: str,
        path_to_vocab_file: str,
        smilestokenizer: BaseTokenizerTemplate,
        splitter: BaseSplitterFactory.splitter,
        label_column_name: str = None,
        model_type="bert",
        run_name=None,
        gradient_accumulation_steps=1,
        mask_prob=0.15,
        learning_rate=5e-5,
        test_size=0.2,
        eval_size=0.1,
        hidden_dropout=0.2,
        eval_steps: int = 10,
        model_size="small",
        early_stopping_patience=None,
        early_stopping_threshold=0.0,
        test_mode=False,
        verbose=True,
        num_train_epochs=10,
        num_train_steps=None,
        scale_target=True,
        max_length=256,
        batch_size=16,
        auto_find_batch_size=True,
        fp16=False,
        warmup_ratio=0.03,
        additional_features=[],
        n_layers=3,
        n_labels=1,
        layer_width=2400,
        logging_steps=10,
        checkpoint_number=None,
        save_total_limit=None,
        max_grad_norm=1.0,
        weight_decay=0.0,
        n_trials=None,
        random_state=None,
        save_dataset_path=None,
        load_dataset_path=None,
        params={},
        skip_training=False,
        weight_decay=0.0,
        max_grad_norm=1.0,
        label_smoothing=0.0,
        freeze_encoder_steps=None,
        gradual_unfreezing=False,
        layerdrop_prob=None,
        stochastic_depth_prob=None,
        *args,
        **kwargs,
    ):
        self.model_factory = model_factory
        self.path_to_outputs = path_to_outputs
        self.dataset_folder = dataset_folder
        self.path_to_vocab_file = path_to_vocab_file
        self.dataset_factory = dataset_factory
        self.features_column_name = features_column_name
        self.label_column_name = label_column_name
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.run_name = run_name
        self.num_train_epochs = num_train_epochs
        self.num_train_steps = num_train_steps
        self.mask_prob = mask_prob
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.eval_size = eval_size
        self.test_mode = test_mode
        self.eval_steps = eval_steps
        self.model_size = model_size
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.verbose = verbose
        self.kwargs = kwargs
        self.path_to_test_data = None
        self.hidden_dropout = hidden_dropout
        self.scale_target = scale_target
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.batch_size = batch_size
        self.auto_find_batch_size = auto_find_batch_size
        self.fp16 = fp16
        self.warmup_ratio = warmup_ratio
        self.additional_features = additional_features
        self.n_layers = n_layers
        self.n_labels = n_labels
        self.layer_width = layer_width
        self.logging_steps = logging_steps
        self.checkpoint_number = checkpoint_number
        self.save_total_limit = save_total_limit
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.n_trials = n_trials
        self.random_state = random_state
        self.save_dataset_path = save_dataset_path
        self.load_dataset_path = load_dataset_path
        self.params = params
        self.skip_training=skip_training
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.label_smoothing = label_smoothing
        self.freeze_encoder_steps = freeze_encoder_steps
        self.gradual_unfreezing = gradual_unfreezing
        self.layerdrop_prob = layerdrop_prob
        self.stochastic_depth_prob = stochastic_depth_prob

        self.run = wandb.init(
            project="Smiles_CGR_transformer",
            mode="disabled" if self.test_mode else "online",
            reinit=False,
            config=params,
        )

        self.define_tokenizer(smilestokenizer)

        # assert isinstance(self.smilestokenizer, BaseTokenizerTemplate), self.tokenizer_not_correct_error

        self.train_test_split = splitter

        if num_train_steps is None:
            self.num_train_steps = -1
        assert (num_train_epochs is not None) or (
            num_train_steps != -1
        ), "Please provide either the number of training steps or the number of training epochs. Note: you cannot provide both."

        if self.num_train_steps is None:
            self.num_train_steps = -1

        self.output_dir = path_finder(
            prefix=self.path_to_outputs,
            path_from_source=f"finetuning/{self.model_type}/{self.model_size}/{self.tokenizer_kind}/{self.run.name}",
            is_file=False,
            create_path_if_unavailable=True,
        )
        self.dataset_factory = self.dataset_factory(
            self.smilestokenizer,
            self.dataset_folder,
            max_length,
            self.output_dir,
            save_dataset_path=self.save_dataset_path,
            load_dataset_path=self.load_dataset_path,
            additional_features=self.additional_features,
            verbose=self.verbose,
            params=self.params,
        )
        if params:
            print("-----------------------------------------")
            print("RECIEVED PARAMETERS:")
            print(json.dumps(params, indent=4))
            print("-----------------------------------------")

    def create(self, dataset: pd.DataFrame, skip_training=False):
        """
        Entry point for creating and configuring a hugging face Trainer.
        This method calls the abstract method create_and_train, that has to be implemented in the child class.
        This method also handles the displayed information.
        self.train_start_message() and self.train_end_message() are called to display the start and end of the training process.
        Args:
            dataset (pd.DataFrame): The dataset to be used for training.
            skip_training (bool, optional): Whether or not to skip the training process. Only set to True for evaluation
        Returns:
            Trainer: A hugging face Trainer object.
        """
        self.train_start_message()
        trainer = self.create_and_train(dataset)
        if not skip_training:
            self.train_end_message()
        return trainer

    @abstractmethod
    def create_and_train(self, dataset: pd.DataFrame, skip_training=False):
        """
        Abstract method to create and configure a hugging face Trainer.
        """
        pass

    def last_checkpoint(self):
        """
        Get the path to the last checkpoint.

        Returns:
            str: The path to the last checkpoint.
        """

        if self.run_name is None:
            return None

        file_list = os.listdir(self.pretrained_model_path)
        pattern = re.compile(r"\d+$")

        numbers = [
            int(pattern.search(filename).group())
            for filename in file_list
            if pattern.search(filename)
        ]

        highest_number = max(numbers)

        return highest_number

    def latest_run_handler(self, run_name, read=False):
        """
        Handle the latest run and related information in a file

        Args:
            run_name (str): The name of the run
            read (bool): If True, reads the latest run. Defaults to False.

        Returns:

        """
        registry_path = path_finder(
            prefix=self.path_to_outputs,
            path_from_source="/latest_model_registey.json",
            is_file=True,
        )

        if not os.path.exists(registry_path):
            with open(
                registry_path,
                "w",
            ) as f:
                json.dump({}, f)
        file_path = registry_path

        try:
            with open(file_path, "r+") as f:
                latest = json.load(f)
        except json.JSONDecodeError:
            latest = {}

        if read:
            return latest[self.model_type][self.model_size][self.tokenizer_kind]

        try:
            latest[self.model_type][self.model_size][self.tokenizer_kind] = run_name
        except KeyError:
            latest.setdefault(self.model_type, {}).setdefault(self.model_size, {})[
                self.tokenizer_kind
            ] = run_name

        with open(file_path, "w") as f:
            json.dump(latest, f)

    def get_checkpoint_folder(self):
        """
        Get the path to the checkpoint folder.

        Returns:
            str: The path to the checkpoint folder.
        """
        checkpoint_folder = None
        if self.checkpoint_number == "last":
            self.checkpoint_number = self.last_checkpoint()
        if self.checkpoint_number is not None:
            checkpoint_folder = os.path.join(
                self.pretrained_model_path,
                f"checkpoint-{self.checkpoint_number}/",
            )
        return checkpoint_folder

    def train_start_message(self):
        """
        Get the start message for training.

        Returns:
            str: The start message for training.
        """
        if self.verbose:
            print("################################################################")
            self.start = timer()
            print(
                f"Training {self.model_type} model of size {self.model_size} with {self.tokenizer_kind} tokenizer on {self.dataset_name} dataset has started."
            )

    def train_end_message(self):
        """
        Get the start message for training.

        Returns:
            str: The start message for training.
        """
        if self.verbose:
            end = timer()
            print(
                f"End of training! Training took {timedelta(seconds=end-self.start)} seconds."
            )
            print("################################################################")

    @abstractmethod
    def define_tokenizer(self, smilestokenizer):
        pass

    def check_eval_length(self, X_eval):
        assert (
            len(X_eval) < 10000
        ), f"The evaluation set is too large (size: {len(X_eval)}). Please reduce the size of the evaluation set to less than 5000 samples. This is to prevent the evaluation set from being too large and causing memory and performance issues. If you need to evaluate on a larger set, please use the test set instead."
        if len(X_eval) > 1000:
            warnings.warn(
                f"Warning: The evaluation set is pretty big (size: {len(X_eval)}), which will slow down considerably the calculations. You should take a smaller eval set size!"
            )
