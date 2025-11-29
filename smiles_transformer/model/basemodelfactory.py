from abc import ABC, abstractmethod

from datasets import Dataset

import json
import importlib.resources as pkg_resources
from smiles_transformer import configurations


class BaseModelFactory(ABC):
    """Abstract base class for creating machine learning models.

    This class provides a template for creating and configuring machine learning models,
    specifically designed for use with datasets and various training configurations.

    Attributes:
        model_type (str): Type of the model to be created.
        dataset (Dataset): The dataset object to be used for training and evaluation.
        eval_steps (int): Number of steps for evaluation.
        num_train_epochs (int): Number of training epochs.
        num_train_steps (int): Number of training steps. Default is -1.
        pretrained_model_path (str): Path to a pretrained model, if applicable.
        early_stopping_patience (int): Number of steps for early stopping patience.
        early_stopping_threshold (float): Threshold for early stopping.
        learning_rate (float): Learning rate for training the model.
        mask_prob (float): Probability of masking for masked language models.
        max_length (int): Maximum sequence length for the model.
        model_size (str): Size of the model (e.g., 'small', 'medium', 'large').
        hidden_dropout (float): Dropout rate for hidden layers in the model.
        warmup_ratio (float): Warmup ratio for learning rate scheduling.
        additional_features (dict): Additional features to be added to the model.
        layer_width (int): Width of the hidden layers in the model.
        n_layers (int): Number of hidden layers in the model.
        n_labels (int): Number of labels as output for the model.
        gradient_accumulation_steps (int): Number of steps for gradient accumulation.
        batch_size (int): Batch size for training.
        auto_find_batch_size (bool): Whether to automatically find an optimal batch size.
        fp16 (bool): Whether to use mixed precision (FP16) training.
        logging_steps (int): Frequency of logging training progress.
        save_total_limit (int): Maximum number of saved checkpoints.
        max_grad_norm (float): Maximum gradient norm for gradient clipping. Defaults to 1.0.
        weight_decay (float): Weight decay coefficient for L2 regularization. Defaults to 0.0.
        output_dir (str): Directory for saving model outputs and checkpoints.
        smilestokenizer: Smiles tokenizer used for processing the input data. This needs to be an instance of the SmilesTokenizer class.
        init_before_create_error (str): Error message for incorrect initialization sequence.
        random_state (int): Random state for reproducibility. Defaults to None, meaning non-reproducible and random.

    Raises:
        NotImplementedError: If `create` or `init_model` methods are not implemented in the subclass.
    """

    init_before_create_error = (
        "the init_model can only be called AFTER the create method"
    )
    no_tokenizer_error = "No tokenizer was given, yet this subclass needs an instantiated smiles tokenizer to be given."
    tokenizer_not_instance = "The given tokenizer is not an instance of the SmilesTokenizer class. Please give an instance, not a class."

    def __init__(
        self,
        output_dir: str,
        model_type: str,
        dataset: Dataset,
        eval_steps: int,
        num_train_epochs: int,
        smilestokenizer=None,
        num_train_steps: int = -1,
        pretrained_model_path=None,
        early_stopping_patience=None,
        early_stopping_threshold=0.0,
        learning_rate=1e-5,
        mask_prob=0.15,
        max_length=512,
        model_size="small",
        hidden_dropout=0.1,
        warmup_ratio=0.03,
        additional_features=[],
        layer_width=2400,
        n_layers=3,
        n_labels=1,
        gradient_accumulation_steps=1,
        batch_size=16,
        auto_find_batch_size=True,
        fp16=False,
        logging_steps=10,
        save_total_limit=None,
        max_grad_norm=1.0,
        weight_decay=0.0,
        random_state=None,
    ):
        self.model_type = model_type
        self.dataset = dataset
        self.eval_steps = eval_steps
        self.num_train_epochs = num_train_epochs
        self.num_train_steps = num_train_steps
        self.pretrained_model_path = pretrained_model_path
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.learning_rate = learning_rate
        self.mask_prob = mask_prob
        self.max_length = max_length
        self.model_size = model_size
        self.hidden_dropout = hidden_dropout
        self.warmup_ratio = warmup_ratio
        self.additional_features = additional_features
        self.layer_width = layer_width
        self.n_layers = n_layers
        self.n_labels = n_labels
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.batch_size = batch_size
        self.auto_find_batch_size = auto_find_batch_size
        self.fp16 = fp16
        self.logging_steps = logging_steps
        self.save_total_limit = save_total_limit
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay
        self.output_dir = output_dir
        self.smilestokenizer = smilestokenizer
        self.random_state = random_state
        assert self.smilestokenizer is not None, self.no_tokenizer_error

        # assert isinstance(self.smilestokenizer, BaseTokenizerTemplate), self.tokenizer_not_instance

        # Open the JSON file as a resource
        with pkg_resources.open_text(configurations, "model_sizes.json") as f:
            model_config = json.load(f)[self.model_type]
        self.model_config = model_config[self.model_size]

    @abstractmethod
    def create(self):
        """
        Abstract method to create and configure a model.

        Subclasses should implement this method to define the process of creating
        and configuring a specific model type.
        """
        pass

    @abstractmethod
    def init_model():
        """
        Abstract method to initialize the model.

        Subclasses should implement this method to define the initialization
        process for the model.
        """
        pass

    def reinit(self):
        """
        Reinitialize the model. This function reinitializes the model and returns a trainer object.
        This methods needs to be called AFTER the load method.
        """
        return self.init_model()
