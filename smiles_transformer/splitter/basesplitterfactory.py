from abc import ABC, abstractmethod
import pandas as pd
from typing import List


class BaseSplitterFactory(ABC):
    """
    Abstract base class for splitter factories. Defines the interface for creating splitters and labelers.
    """

    def __init__(self, random_state: int):
        """
        Initializes the splitter factory with a random state for reproducibility.

        Args:
            random_state: The random state to use.
        """
        self.random_state = random_state

    @abstractmethod
    def create(self) -> tuple:
        """
        Creates and returns the labeler and splitter functions.

        Returns:
            tuple: A tuple containing the labeler and splitter functions.
        """

        pass

    @abstractmethod
    def splitter(
        self,
        X: pd.DataFrame,
        y: pd.Series | None,
        splits: List[str] = ["train", "test", "eval"],
        test_size: float = 0.25,
        remove_split_column: bool = False,
    ) -> List:
        """
        Splits data into specified splits using a specific implementation.

        Args:
            X: Data to be split.
            y: Target labels (optional).
            splits: Desired split types.
            test_size: Proportion for test split.
            remove_split_column: Whether to remove the split column.

        Returns:
            list: A list containing the split DataFrames and Series, in the order of splits.
        """
        pass

    @abstractmethod
    def labeler(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        test_size: float = 0.25,
        eval_size: float = 0.25,
    ) -> tuple:
        """
        Assigns labels to data for training, testing, and evaluation splits.

        Args:
            X: Data to be labeled.
            y: Target labels (optional).
            test_size: Proportion for test split.
            eval_size: Proportion for evaluation split.

        Returns:
            tuple: A tuple containing the labeled DataFrames or Series for X and y.
        """
        pass
