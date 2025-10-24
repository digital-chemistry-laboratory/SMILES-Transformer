import json
from abc import ABC, abstractmethod

import pandas as pd
from datasets import DatasetDict, DatasetInfo

from smiles_transformer.tokenizer.basetokenizertemplate import BaseTokenizerTemplate
from smiles_transformer.utils.path_finder import path_finder


class BaseDatasetFactory(ABC):
    """
    Abstract class for creating datasets.

    Attributes:
        smilestokenizer (BaseTokenizerTemplate): A tokenizer for SMILES strings. This needs to be an instance of the SmilesTokenizer class.
        data_folder_path (str): The folder path where the dataset is stored.
        max_length (int): Maximum length of SMILES strings.
        tokenizer_kind (str): Kind of tokenizer to be used.
        vocab_path (str): Path to the vocabulary file.
        output_dir (str): Path to the output directory.
        save_dataset_path (str): Path to save the preprocessed dataset.
        verbose (bool): Whether to print information during preprocessing.
        params (dict): the entire configuration dictionary.
    Methods:
        create(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame): Abstract method to create and configure a dataset.
        encode_dataset(dataset: DatasetDict): Encode a dataset.
    """

    no_output_dir_error = "No output directory specified. Please specify an output directory during instantiation of the class."
    tokenizer_not_instance = "The given tokenizer is not an instance of the SmilesTokenizer class. Please give an instance, not a class."

    def __init__(
        self,
        smilestokenizer: BaseTokenizerTemplate,
        data_folder_path: str,
        max_length: int,
        output_dir: str = None,
        save_dataset_path=None,
        load_dataset_path=None,
        additional_features=[],
        verbose=True,
        params={},
    ):
        self.output_dir = output_dir
        self.smilestokenizer = smilestokenizer
        self.data_folder_path = data_folder_path
        self.max_length = max_length
        self.save_dataset_path = save_dataset_path
        self.load_dataset_path = load_dataset_path
        self.additional_features = additional_features
        self.verbose = verbose

        self.params = params

        # assert isinstance(smilestokenizer, BaseTokenizerTemplate), self.tokenizer_not_instance #TODO: Fix this

    @abstractmethod
    def create(
        self,
        X_train,
        y_train,
        X_eval: pd.DataFrame = None,
        y_eval: pd.DataFrame = None,
        X_test=None,
        y_test=None,
        scale_target: bool = True,
    ):
        pass

    def encode_dataset(self, dataset: DatasetDict):
        """
        Encodes a dataset using a SMILES tokenizer.

        This method iterates over each split in the dataset and applies the SMILES tokenizer to the text data.
        The text data is encoded with padding up to the maximum length specified by `self.max_length`.

        Args:
            dataset (DatasetDict): A dictionary of datasets, where each key is a split of the dataset
                                (e.g., 'train', 'test') and the value is the dataset for that split.

        Returns:
            DatasetDict: The modified dataset with each text entry encoded using the SMILES tokenizer.
                        The original text columns are removed after encoding.

        Example:
            >>> dataset = {"train": train_dataset, "test": test_dataset}
            >>> encoded_dataset = encode_dataset(dataset)
            >>> print(encoded_dataset['train'][0])

        Note:
            This method relies on `self.smilestokenizer` to encode the text and `self.max_length` to define the
            maximum encoding length. The function assumes these attributes are already defined in the class.

        """

        for split in dataset:
            dataset[split] = dataset[split].map(
                lambda x: self.smilestokenizer.batch_encode_plus(
                    x,
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True,
                ),
                batched=True,
                input_columns=["text"],
                remove_columns=["text"],
            )
            if self.additional_features:
                dataset[split] = dataset[split].map(self.add_features)
                dataset[split] = dataset[split].remove_columns(self.additional_features)
        if self.save_dataset_path:
            self.save_dataset(dataset)

        return dataset

    def add_features(self, line):
        """
        Adds features to the input_ids line.
        Args:
            line (dict): A dictionary of the dataset

        Returns:
            list: A list of features added, and the input ids.
        """
        feat_list = []
        for feature in self.additional_features:
            feat_list = feat_list + [line[feature]]
        line["additional_features"] = feat_list
        return line

    def save_dataset(self, dataset_dict):
        """
        Save the dataset to disk.

        Args:
            dataset (DatasetDict): The dataset to save.
        """

        path = path_finder(
            prefix=self.data_folder_path,
            path_from_source=self.save_dataset_path,
            is_file=True,
            create_path_if_unavailable=True,
        )
        if self.verbose:
            print("-----------------------------------------")
            print(f"Saving dataset to {path}")
        dataset_dict.save_to_disk(path)
        if self.verbose:
            print("Dataset saved.")
            print("-----------------------------------------")

    def create_dataset_info(self, X):
        self.params["split_size"] = len(X)
        preprocessing_info = json.dumps(
            self.params, indent=4
        )  # Converts dict to a formatted string
        # Combine split size and preprocessing info into a description

        # Create additional info
        info = DatasetInfo(description=preprocessing_info)

        return info
