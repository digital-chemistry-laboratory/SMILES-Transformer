import os
from typing import List

import numpy as np
from joblib import Parallel, delayed

from smiles_transformer.utils.path_finder import path_finder
from smiles_transformer.tokenizer.basetokenizertemplate import BaseTokenizerTemplate

from pathlib import Path


class Vocab:
    """Class for managing a vocabulary file.

    Attributes:
        vocab_file_path (str): Path to the vocabulary file.
        vocab_set (set): Set to store the vocabulary.

    Methods:
        verify_vocab_file(): Creates the vocabulary file if it doesn't exist.
        add_tokens_to_vocab(tokens: list) -> None: Adds tokens to the vocabulary.
        _flatten(nested_list: list) -> list: Flattens a nested list of tokens.
        purge_vocab(): Deletes the vocabulary file.
    """  # noqa 501

    vocab_size_too_small = "The max_vocab_size must be greater than 0."

    def __init__(
        self,
        smilestokenizer: BaseTokenizerTemplate,
        path_to_vocab_folder: str,
        # tokenizer_kind: str,
        sampling: bool = False,
        sample_size: int = 1000,
        max_vocab_size: int = 200,
        min_frequency: int = 3,
    ):
        """Initializes a new instance of the Vocab class. Runs the verify_vocab_file method.

        Args:
            smilestokenizer (BaseTokenizerTemplate): A tokenizer for SMILES strings.
            path_to_vocab_folder (str): Path to the folder containing the vocabulary file.
            sampling (bool): If True, the vocabulary is created from a random sample of the SMILES strings. Default: False.
            sample_size (int): Number of SMILES strings to sample. if sampling=True Default: 1000.
            max_vocab_size (int): Max size of the vocabulary if the tokenizer uses BPE. Default: 200.
            min_frequency (int): Minimum frequency of a token to be included in the vocabulary if the tokenizer uses BPE. Default: 3.
        """  # noqa 501
        assert max_vocab_size > 0, self.vocab_size_too_small
        self.tokenizer_kind = smilestokenizer.tokenizer_kind
        self.path_to_vocab_folder = path_to_vocab_folder
        self.path_to_vocab_file = (
            Path(path_to_vocab_folder) / f"vocab_{self.tokenizer_kind}.txt"
        )
        self.verify_vocab_file()

        self.smilestokenizer = smilestokenizer(vocab_file=self.get_vocab_file_path())
        # self.tokenizer_kind = self.smilestokenizer.tokenizer_kind

        self.sampling = sampling
        self.sampling_size = sample_size
        self.vocab_set = set()

        self.vocab_size = max_vocab_size
        self.min_frequency = min_frequency

    def autogenerate(
        self,
        smiles: List[str] | np.ndarray,
        from_scratch: bool = False,
    ):
        """
        Generates a vocabulary from a list of SMILES strings.

        Args:
        smiles (list/np.ndarray): List or numpy array of SMILES strings.
        from_scratch (bool): If True, the vocabulary is deleted and created from scratch. Default: False.

        Returns:
        None
        """  # noqa 501

        if from_scratch:
            self.from_scratch()
        if self.sampling and len(smiles) > self.sampling_size:
            smiles = np.random.choice(
                smiles.flatten(), self.sampling_size, replace=False
            )
        if self.tokenizer_kind == "bpe":
            self.smilestokenizer.train(
                smiles_list=smiles,
                vocab_size=self.vocab_size,
                min_frequency=self.min_frequency,
            )
            return
        basic_tokenizer = self.smilestokenizer.base_tokenizer
        new_vocab = Parallel(n_jobs=-1)(delayed(basic_tokenizer)(x) for x in smiles)
        self.add_tokens_to_vocab(new_vocab)

    def get_vocab_file_path(self) -> str:
        """Returns the path to the vocabulary file.

        Returns:
            The path to the vocabulary file.
        """
        return path_finder(
            prefix=self.path_to_vocab_folder,
            path_from_source=f"vocab_{self.tokenizer_kind}.txt",
            is_file=True,
        )

    def read_vocab(self) -> list:
        """Reads the vocabulary file.

        Returns:
            A list of tokens in the vocabulary.
        """
        with open(self.get_vocab_file_path(), "r") as f:
            return f.read().splitlines()

    def verify_vocab_file(self):
        """Creates the vocabulary file if it doesn't exist."""
        if not os.path.exists(self.get_vocab_file_path()):
            try:
                print(f"{self.get_vocab_file_path()} file not found. Creating it...")
                with open(
                    self.get_vocab_file_path(),
                    "w",
                ) as f:
                    f.write(
                        "\n".join(
                            [
                                "[PAD]",
                                "[UNK]",
                                "[CLS]",
                                "[SEP]",
                                "[MASK]",
                            ]
                        )
                    )
            except FileNotFoundError:
                raise Exception(
                    f"{os.path.abspath(self.get_vocab_file_path())} path not found. Please verify the path. If the file does not exist, it should create it for you, this is a path issue."  # noqa 501
                )

    def add_tokens_to_vocab(self, tokens: list) -> None:
        """Adds tokens to the vocabulary file.

        Args:
            tokens (list): List of tokens to add to the vocabulary.

        Returns:
            None
        """

        tokens = self._flatten(tokens)
        with open(
            self.get_vocab_file_path(),
            "r+",
        ) as f:
            self.vocab_set.update(f.read().splitlines())
            new_vocab = list(self.vocab_set.union(tokens))
            # Move special tokens to the beginning of the vocab list
            special_tokens = [t for t in new_vocab if t not in tokens]
            other_tokens = [t for t in new_vocab if t in tokens]

            sorted_vocab = special_tokens + other_tokens
            f.seek(0)
            f.write("\n".join(sorted_vocab))
            f.truncate()
        self.vocab_set.clear()

    def _flatten(self, nested_list: list) -> list:
        """Flattens a nested list of tokens.

        Args:
            nested_list (list): Nested list of tokens to flatten.

        Returns:
            A flattened list of tokens.
        """
        flattened_list = []
        for item in nested_list:
            if isinstance(item, list):
                flattened_list.extend(self._flatten(item))
            else:
                flattened_list.append(item)
        return flattened_list

    def purge_vocab(self):
        """
        Deletes the vocabulary file if it exists.

        Prints:
            "{vocab} file not found" message: If the vocabulary file is not found.
        """  # noqa 501
        if not os.path.exists(self.get_vocab_file_path()):
            print(f"{self.get_vocab_file_path()}: file does not exist.")
            return
        os.remove(self.get_vocab_file_path())

    def from_scratch(self):
        """Deletes the vocabulary file and creates a new one."""
        self.purge_vocab()
        self.verify_vocab_file()

    @property
    def size(self) -> int:
        """Returns the size of the vocabulary.

        Returns:
            An integer representing the size of the vocabulary.
        """
        return len(self.read_vocab())
