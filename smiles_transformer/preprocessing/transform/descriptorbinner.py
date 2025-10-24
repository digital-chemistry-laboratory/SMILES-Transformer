import ast
import pandas as pd
import numpy as np
import numbers
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer
from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
from tqdm import tqdm


class DescriptorBinner(TransformTemplate):
    """
    Bins molecular descriptors into discrete categories based on the specified strategy.

    This class processes molecular descriptors from a pandas DataFrame and converts them
    into discrete bins using specified binning strategies (e.g., uniform, quantile, k-means).

    Attributes:
        descriptors_to_bin (list[str]): List of column names of descriptors to be binned.
        num_bins (int): Number of bins to divide the data into.
        bin_encode (str): Encoding method for the bins. Default is "ordinal".
        strategy (str): Binning strategy to use. Options: 'uniform', 'quantile', 'kmeans'.
        bins_to_numbers (bool): If True, bins are represented as numbers. Otherwise, prefixed strings. Default: False

    Methods:
        step(batch): Applies the binning transformation to the input DataFrame.
    """

    def _binner_parameters(self):
        self.descriptors_to_bin = self.kwargs.get("descriptors_to_bin")
        self.n_bins = self.kwargs.get("num_bins")
        # bin_encode is only properly implemented for ordinal!
        self.bin_encode = "ordinal"
        self.strategy = self.kwargs.get("binning_strategy")
        self.bins_to_numbers = self.kwargs.get("bins_to_numbers", False)

    def step(self, batch):
        """
        Convert SMILES to SMILES with molecular descriptors.
        """

        self._binner_parameters()

        batch = self.bin_descriptors(
            df=batch,
            selected_descriptors=self.descriptors_to_bin,
            n_bins=self.n_bins,
            encode=self.bin_encode,
            strategy=self.strategy,
        )
        return batch

    def addendum_numeri_composer(self, num: int):
        if self.bins_to_numbers:
            return num
        return "BIN_" + str(num)

    def transform_scalars(self, num: int):
        # Directly apply the function to the scalar
        return self.addendum_numeri_composer(num)

    def transform_1d_array(self, array):
        # Apply the function to each element in a 1D array
        return [self.addendum_numeri_composer(x) for x in array]

    def transform_2d_array(self, matrix):
        # Apply the function to each element in a 2D array
        return [
            [self.addendum_numeri_composer(value) for value in row] for row in matrix
        ]

    def bin_Series_of_matrices(
        self, data, column="matrix_descriptors", strategy="uniform"
    ):
        """
        Bin data in a pd.Series of square matrices according to the specified strategy.

        Parameters:
            data (pd.DataFrame): DataFrame containing a column with n x n matrices.
            column (str): The column name in the DataFrame that contains the matrices to be binned.
            strategy (str): The binning strategy, 'uniform', 'quantile', or 'kmeans'.

        Returns:
            pd.DataFrame: The original DataFrame with an added column for the binned data.
        """
        # Step 1: Flatten matrices to a 1D array after ensuring they are numpy arrays
        flat_list = np.concatenate([np.asarray(mat).flatten() for mat in data[column]])

        # Determine the strategy and apply binning
        if strategy == "kmeans":
            km = KMeans(n_clusters=self.n_bins, random_state=0)
            km.fit(flat_list.reshape(-1, 1))
            binned_indices = km.predict(flat_list.reshape(-1, 1))
        elif strategy == "uniform":
            bins = np.linspace(flat_list.min(), flat_list.max(), self.n_bins + 1)
            binned_indices = np.digitize(flat_list, bins) - 1
        elif strategy == "quantile":

            if len(np.unique(flat_list)) == 1:
                quantiles = np.ones(len(flat_list), dtype=int)
            else:
                quantiles = pd.qcut(
                    flat_list, q=self.n_bins, labels=False, duplicates="drop"
                )
            binned_indices = quantiles
        else:
            raise ValueError(
                "Unsupported binning strategy. Choose from 'uniform', 'quantile', or 'kmeans'."
            )

        # Reconstruct the flattened binned indices back to the original matrix structure
        binned_matrices = []
        start_index = 0
        for matrix in data[column]:
            size = np.asarray(matrix).size
            new_matrix = binned_indices[start_index : start_index + size].reshape(
                np.asarray(matrix).shape
            )
            binned_matrices.append(new_matrix)
            start_index += size

        return binned_matrices

    def bin_Series_of_lists(self, data, column, strategy="uniform"):
        """
        Bin data in a pd.Series of lists according to the specified strategy.

        Parameters:
            data (pd.DataFrame): DataFrame containing a column with lists of numeric values.
            column (str): The column name in the DataFrame that contains the lists to be binned.
            strategy (str): The binning strategy, 'uniform', 'quantile', or 'kmeans'.

        Returns:
            pd.DataFrame: The original DataFrame with an added column for the binned data.
        """

        # Step 1: Flatten the list
        flat_list = np.array([item for sublist in data[column] for item in sublist])

        # Determine the strategy and apply binning
        if strategy == "kmeans":
            km = KMeans(n_clusters=self.n_bins, random_state=0)
            km.fit(flat_list.reshape(-1, 1))
            binned_indices = km.predict(flat_list.reshape(-1, 1))
        elif strategy == "uniform":
            bins = np.linspace(flat_list.min(), flat_list.max(), num=self.n_bins + 1)
            binned_indices = np.digitize(flat_list, bins) - 1
        elif strategy == "quantile":
            if len(np.unique(flat_list)) == 1:
                quantiles = np.ones(len(flat_list), dtype=int)
            else:
                quantiles = pd.qcut(
                    flat_list, q=self.n_bins, labels=False, duplicates="drop"
                )  # handle duplicates
            binned_indices = quantiles
        else:
            raise ValueError(
                "Unsupported binning strategy. Choose from 'uniform', 'quantile', or 'kmeans'."
            )
        # Reconstruct to the original list structure
        binned_descriptors = []
        start = 0
        for sublist in data[column]:
            end = start + len(sublist)
            binned_descriptors.append(binned_indices[start:end].tolist())
            start = end

        return binned_descriptors

    def bin_descriptors(
        self,
        df: pd.DataFrame,
        selected_descriptors,
        n_bins=8,
        encode="ordinal",
        strategy="unifrom",
    ):
        """
        Bins all the descriptors specified and returns dataframe with binned descriptors

        Arguments:

        row : pd.Series from a dataframe containing the selected descriptors
        slected_descriptors : list of selected descriptors that should be binned
        n_bins : int or array-like of shape (n_features,), default=5
            The number of bins to produce. Raises ValueError if ``n_bins < 2``.
        encode : {'onehot', 'onehot-dense', 'ordinal'}
            Method used to encode the transformed result. For atom descriptors only ordinal is applied.
        strategy : {'uniform', 'quantile', 'kmeans'}
            Strategy used to define the widths of the bins.
        """
        for selected_descriptor in tqdm(
            selected_descriptors, desc="descriptor columns processed"
        ):
            binner = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
            if isinstance(df[selected_descriptor][0], numbers.Number):
                df[selected_descriptor] = binner.fit_transform(
                    df[selected_descriptor].values.reshape(-1, 1)
                )
                binned_data = df[selected_descriptor].apply(int)
                df[selected_descriptor] = [
                    self.transform_scalars(x) for x in binned_data
                ]
            elif isinstance(
                df[selected_descriptor][0], (list, np.ndarray)
            ) and not isinstance(
                df[selected_descriptor][0][0], (list, np.ndarray)
            ):  # Check if it's 1D array
                binned_data = self.bin_Series_of_lists(
                    df, selected_descriptor, strategy=strategy
                )
                df[selected_descriptor] = [
                    self.transform_1d_array(x) for x in binned_data
                ]
            elif isinstance(
                df[selected_descriptor][0], (list, np.ndarray)
            ) and isinstance(
                df[selected_descriptor][0][0], (list, np.ndarray)
            ):  # Check if 2D array
                binned_data = self.bin_Series_of_matrices(
                    df, selected_descriptor, strategy=strategy
                )
                df[selected_descriptor] = [
                    self.transform_2d_array(matrix) for matrix in binned_data
                ]

            else:
                raise TypeError(
                    f"Your descriptor: {selected_descriptor} of type: {type(selected_descriptor)} in descriptors_to_bin is neither a number, nor a 1D array, nor a 2D array"
                )
        return df

    @property
    def init_message(self) -> str:
        return f"""Binning the selected descriptors."""
