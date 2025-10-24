from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Union, Optional

from .basesplitterfactory import BaseSplitterFactory


class SKLearnSplitter(BaseSplitterFactory):
    """
    This class is used to split the data into training and testing sets.
    It uses Scikit-learn's train_test_split function.
    """

    def create(self) -> Tuple[callable, callable]:
        """
        This method returns the labeler and splitter functions.
        """
        return self.labeler, self.splitter

    def splitter(
        self,
        X,
        y: Optional = None,
        splits: List[str] = ["train", "test", "eval"],
        split_sizes: Dict[str, float] = {"test": 0.25, "eval": 0.1},
    ) -> List[Union[List, None]]:
        """
        Splits data into training, testing, and evaluation sets based on specified sizes.

        Parameters:
            X: The input features to split.
            y: The target variables to split, optional.
            splits: A list indicating which splits to create.
            split_sizes: A dictionary specifying the sizes of test and eval splits.

        Returns:
            A list of datasets split according to the provided specifications. This list
            may contain any combination of X_train, X_test, X_eval, y_train, y_test, and y_eval,
            depending on whether y is provided and the splits required.
        """

        eval_size = split_sizes["eval"]
        test_size = split_sizes["test"]
        if split_sizes["test"] < 1:
            test_size = int(test_size * len(X))
        if split_sizes["eval"] < 1:
            eval_size = int(eval_size * len(X))
        X_train, X_test, X_eval, y_train, y_test, y_eval = (
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if y is not None:
            if eval_size > 0:
                X_train, X_eval, y_train, y_eval = train_test_split(
                    X, y, test_size=eval_size, random_state=self.random_state
                )
            if test_size > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_train,
                    y_train,
                    test_size=test_size,
                    random_state=self.random_state,
                )
            return_list = [X_train, X_test, X_eval, y_train, y_test, y_eval]
        else:

            if eval_size > 0:
                X_train, X_eval = train_test_split(
                    X, test_size=eval_size, random_state=self.random_state
                )
            if test_size > 0:
                X_train, X_test = train_test_split(
                    X_train,
                    test_size=test_size,
                    random_state=self.random_state,
                )
            return_list = [X_train, X_test, X_eval]
        return [x for x in return_list if x is not None]

    def labeler(
        self,
        X,
        y: Optional = None,
        test_size: float = 0.25,
        eval_size: float = 0.1,
    ) -> Union[Tuple, List]:
        """
        A placeholder function for potential preprocessing or labeling, currently returns inputs unmodified.

        Parameters:
            X: The input features.
            y: The target variables, optional.
            test_size: The size of the test split, not used in current implementation.
            eval_size: The size of the eval split, not used in current implementation.

        Returns:
            The unmodified input features, and target variables if provided.
        """
        if y is not None:
            return X, y
        return X
