from abc import ABC, abstractmethod
from datetime import timedelta
from timeit import default_timer as timer

import pandas as pd


class TransformTemplate(ABC):
    def __init__(
        self,
        verbose=True,
        **kwargs,
    ):
        """
        Template class for processing data. This class is used to define the interface for all data processing classes.
        To implement this class, the user must implement the step method and the init_message property method.
        This can be a transform method (transforms CGR to SMILES/CGR), or a filter method (remove points with confidence below 0.5).

        Args:
            verbose (bool): Whether to print the progress of the transform.
            kwargs: Additional arguments to pass to the hook method. This can be used to pass specific arguments to the transforms,
            taking into account and additional parameters that need to be set.
        """
        self.verbose = verbose
        self.index = 0
        self.filter_too_strict_warning = False
        self.kwargs = kwargs

    def transform(
        self,
        batch: pd.DataFrame,
        in_column: str,
        out_column: str = None,
    ) -> pd.DataFrame:
        """
        transforms the batch based on the provided criteria.
        This is the method that needs to be called to transform the data.

        Args:
            batch (pd.DataFrame): The batch of data to filter.
            in_column (str): The name of the input column.
            out_column (str): The name of the output column. If the transform is a filter, out_column is not needed.

        Returns:
            pd.DataFrame: The filtered batch of data.

        """
        
        if self.verbose:
            self.print_beginning_message()
        self.in_column = in_column
        self.out_column = out_column
        filtered_batch = self.step(batch)
        filtered_batch = self.remove_empty_lines(filtered_batch)
        
        if self.verbose:
            print(f"Removed {len(batch) - len(filtered_batch)} of {len(batch)} samples, resulting in {len(filtered_batch)} samples.")
            self.print_end_message()

        return filtered_batch

    @abstractmethod
    def step(self, batch: pd.DataFrame) -> pd.DataFrame:
        """
        The step method. This method is used to implement the processing logic.
        This method must be implemented by the subclass.

        Args:
            batch (pd.DataFrame): The batch of data to process.

        Returns:
            pd.DataFrame: The processed batch of data.
        """
        pass

    def remove_empty_lines(self, filtered_batch) -> pd.DataFrame:
        """
        Removes empty lines from the filtered set.

        Args:
            filtered_batch (pd.DataFrame): The filtered batch of data.

        Returns:
            DataFrame: The filtered batch of data with empty lines removed.
        """
        return filtered_batch[filtered_batch[self.in_column] != ""].reset_index(
            drop=True
        )

    def print_beginning_message(self):
        """
        Prints a message at the beginning of the transform.

        Args:
            message (str): The message to print.
        """
        print("-----------------------------------------")
        print(f"Starting transform:\n{self.init_message}")
        self.start = timer()

    def print_end_message(self):
        """
        Prints a message at the end of the transform.
        """
        end = timer()
        print(f"Transform finished in {timedelta(seconds=end - self.start)}")
        print("-----------------------------------------")

    @property
    @abstractmethod
    def init_message(self) -> str:
        """
        Message describing the transform.
        This property must be implemented by the subclass.
        """
        pass
