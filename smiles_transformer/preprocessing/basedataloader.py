import os

import numpy as np
import pandas as pd

from smiles_transformer.preprocessing.transform import TransformTemplate
from smiles_transformer.utils.path_finder import path_finder
from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    """
    Abstract base class for data loading that supports preprocessing transformations and dataset splitting.

    This class is designed to initialize a data loader for a specific dataset, applying a series of transformations
    and handling dataset splitting into training, testing, and evaluation sets as needed.

    Attributes:
        path: The directory path where the dataset is located.
        dataset_name: The name of the dataset file.
        transforms: A list of transformation objects to apply to the dataset.
        labeler: A labeling function from BaseSplitterFactory for data points.
        n_points: The number of data points to load; if None, all data points are loaded.
        verbose: Enables verbose output if set to True.
        shuffle: If True, shuffles the dataset before processing.
        test_size: The proportion of the dataset to be used as a test set; if None, no test split is made.
        eval_size: The proportion of the dataset to be used as an evaluation set; if None, no evaluation split is made.
        kwargs: Additional keyword arguments passed to transformation processes.

    Raises:
        TypeError: If the transformations provided are not instances of TransformTemplate.
    """

    def __init__(
        self,
        path: str,
        dataset_name: str,
        transforms: list[TransformTemplate],
        labeler: callable,
        n_points: int | None = None,
        verbose: bool = True,
        shuffle: bool = True,
        test_size: float | None = None,
        eval_size: float | None = None,
        original_column_name: str = None,
        **kwargs,
    ):
        self.path = path
        self.dataset_name = dataset_name

        self.transforms = []
        self.test_size = test_size
        self.eval_size = eval_size
        self.dataset_name = dataset_name

        self.original_column_name = original_column_name
        for process in transforms:
            self.transforms.append(process(verbose, **kwargs))
        if len(transforms):
            if not isinstance(self.transforms[0], TransformTemplate):
                raise TypeError(
                    f"transform must be a TransformTemplate or a list of TransformTemplate, which is not the case for {type(self.transforms[0])}"
                )

        self.file = path_finder(path, dataset_name, is_file=True)
        self.labeler = labeler
        self.verbose = verbose
        self.shuffle = shuffle
        self.n_points = n_points
        self.index = 0
        self.end_of_batch = False
        self.kwargs = kwargs

    @abstractmethod
    def load_dataset(
        self,
        in_column_list: str,
        out_column_list: str,
        separator: str = "\t",
        # additional_columns: list[str] = [],
    ) -> np.ndarray:
        """
        Loads the dataset.

        This is an abstract method that must be implemented by subclasses to specify how the dataset is loaded.

        Parameters:
            main_column_name: The primary column name to focus on during loading.
            separator: The delimiter used in the dataset file, defaults to a tab character.

        Returns:
            A numpy array containing the loaded dataset.
        """
        pass

    def process(
        self, batch: pd.DataFrame, in_column_list: str, out_column_list: str
    ) -> pd.DataFrame:
        """
        Applies the specified transformations to a batch of data.

        Parameters:
            batch: A pandas DataFrame representing the data batch to process.
            in_column_list: The list of input columns to apply transformations to.
            out_column_list: The list of output columns store the transformations in.

        Returns:
            A pandas DataFrame with the applied transformations.
        """
        if len(batch):
            for filter_item, in_column, out_column in zip(
                self.transforms, in_column_list, out_column_list
            ):
                batch = filter_item.transform(
                    batch,
                    in_column=in_column,
                    out_column=out_column,
                )
        return batch
