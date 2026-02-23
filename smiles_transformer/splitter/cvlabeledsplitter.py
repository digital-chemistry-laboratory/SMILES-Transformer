from .basesplitterfactory import BaseSplitterFactory
from typing import List, Tuple, Optional, Union
import pandas as pd


class CVLabeledSplitter(BaseSplitterFactory):
    """Splits data into training, testing, and evaluation sets using a split column. This splitter is used for cross-validation, where splits were not already done before.
    If splits were made before, use the AugmentCVLabeledSplitter.

    Inherits from:
        BaseSplitterFactory

    Attributes:
        no_split_column_error (str): Error message for missing split column.
        not_same_length_error (str): Error message for mismatched X and y lengths.

    Raises:
        ValueError: If the dataset lacks a split column or X and y have different lengths.
    """

    test_size_error = "test_size must be in the format 'n/m', where n is the split number and m is the total number of splits."
    random_state_error = "A random state must be set for the CV splitter."
    len_eval_size_error = (
        "The evaluation size must be less than the number of training samples."
    )

    no_split_column_error = "The dataset must include a column with the split type."
    not_same_length_error = "The length of X and y must be the same."

    def splitter(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        splits: List[str] = ["train", "test", "eval"],
        split_sizes: dict[str, float] = {"test": 0.25, "eval": 0.1},
    ) -> Tuple[List[pd.DataFrame], List[Optional[pd.Series]]]:
        """Splits data into specified splits.

        Args:
            X (pd.DataFrame): Data to be split.
            y (pd.Series, optional): Target labels. Defaults to None.
            splits (list, optional): Desired split types. Defaults to ["train", "test", "eval"].
            split_sizes (dict, optional): Proportions for each split. Defaults to {"test": 0.25, "eval": 0.1}. NOT USED IN THIS SPLITTER, BUT KEPT FOR CONSISTENCY.

        Returns:
            tuple: A tuple containing a list of DataFrames for X and a list of Series for y, in the order of splits.
        """
        if "split" not in X.columns:
            raise ValueError(self.no_split_column_error)
        if y is not None and len(X) != len(y):
            raise ValueError(self.not_same_length_error)
        split_list = []
        for split in splits:
            split_list = split_list + self.get_split(X, y, split)
        X_list = split_list[::2]
        y_list = split_list[1::2]
        return X_list + y_list

    def get_split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        split: str = "train",
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Retrieves a specific split from the data.

        Args:
            X (pd.DataFrame): Data to be split.
            y (pd.Series, optional): Target labels. Defaults to None.
            split (str, optional): Desired split type. Defaults to "train".

        Returns:
            tuple: A tuple containing the DataFrame for X and Series for y of the specified split.
        """

        X_split = X[X["split"] == split]
        if y is not None:
            y_split = y[X["split"] == split]
            X_split, y_split = self.drop_split_labels(X_split, y_split)
            return self.reset_index(X_split, y_split)
        X_split = self.drop_split_labels(X_split)
        return self.reset_index(X_split)

    def drop_split_labels(
        self, *iterables: Union[pd.DataFrame, pd.Series]
    ) -> List[Union[pd.DataFrame, pd.Series]]:
        """Drops the "split" column from provided DataFrames or Series.

        Args:
            *iterables: DataFrames or Series to drop the column from.

        Returns:
            list: A list of DataFrames or Series with the "split" column removed.
        """
        return_list = []
        for iterable in iterables:
            return_list.append(iterable.drop(columns=["split"], errors="ignore"))
        return return_list

    def reset_index(
        self, *iterables: Union[pd.DataFrame, pd.Series]
    ) -> Tuple[Union[pd.DataFrame, pd.Series], ...]:
        """Resets the indices of DataFrames or Series.

        Args:
            *iterables: DataFrames or Series to reset indices.

        Returns:
            list: A list of DataFrames or Series with reset indices.
        """
        return_list = []
        if len(iterables) == 1:
            iterables = iterables[0]
        for iterable in iterables:
            return_list.append(iterable.reset_index(drop=True))
        return return_list

    def labeler(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        test_size: str = None,
        eval_size: float = 0.25,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """

        Args:
            X (pd.DataFrame): Data to be labeled.
            y (pd.Series, optional): Target labels. Defaults to None.
            test_size (str): A string indicating which split it is out of how many. Ex: "2/5" for the second of five splits.
            eval_size (float, optional): Proportion for evaluation split. Defaults to 0.25.

        Returns:
            tuple: A tuple containing the labeled DataFrames or Series for X and y.
        """
        assert "/" in test_size, self.test_size_error
        split_number, total_splits = [int(x) for x in test_size.split("/")]
        data = X
        begin_test_index = int((split_number - 1) * len(data) / total_splits)
        end_test_index = int(split_number * len(data) / total_splits)
        if y is not None:
            data = data.assign(y=y)
        # shuffle the data if it's the first split:
        if begin_test_index == 0:
            data = data.sample(frac=1, random_state=self.random_state)
        data = data.reset_index(drop=True)
        data["split"] = "train"
        data.iloc[begin_test_index:end_test_index, data.columns.get_loc("split")] = (
            "test"
        )
        if eval_size < 1:
            eval_size = int(eval_size * len(data))
        assert eval_size < len(data[data["split"] == "train"]), self.len_eval_size_error
        # randomly assign the value "eval" to eval_size number of points, but the selected points MUST have "train" in their "split" column:
        eval_indices = data[data["split"] == "train"].sample(n=eval_size, random_state=self.random_state).index
        data.iloc[eval_indices, data.columns.get_loc("split")] = "eval"

        data = data.reset_index(drop=True)
        if y is not None:
            return data.drop(columns=["y"]), data[["y", "split"]]
        return data

    def create(self) -> Tuple[callable, callable]:
        """Creates and returns the labeler and splitter functions.

        Returns:
            tuple: A tuple containing the labeler and splitter functions.
        """
        return self.labeler, self.splitter
