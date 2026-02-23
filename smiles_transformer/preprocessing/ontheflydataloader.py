import numpy as np
import pandas as pd


from smiles_transformer.preprocessing.basedataloader import BaseDataLoader


class OnTheFlyDataLoader(BaseDataLoader):

    def load_dataset(
        self,
        in_column_list: str,
        out_column_list: str,
        separator: str = "\t",
    ) -> np.ndarray:
        """
        Loads a dataset from a CSV file, extracts a specified column as a NumPy array, and applies dynamic selection and filtering.

        This method assumes the dataset file path is specified in the `self.file` attribute. It supports dynamic filtering based on various criteria defined elsewhere in the class (e.g., reaction mapping, CGR conversion, and bond change limits).

        Args:
            separator: The character used to separate fields in the CSV file. Defaults to a tab ("\t").

        Returns:
            A NumPy array containing the data from the specified column of the CSV file, after applying the necessary transformations and filters.
        """

        df = pd.read_csv(
            self.file,
            sep=separator,
            on_bad_lines="skip",
        )
        if self.n_points is not None:
            if self.n_points <= 1:
                self.n_points = int(df.shape[0] * self.n_points)

        df["original_input"] = df[self.original_column_name]

        return self.select_points_on_demand(
            X=df,
            in_column_list=in_column_list,
            out_column_list=out_column_list,
        )

    def select_points_on_demand(
        self,
        X: pd.DataFrame,
        in_column_list: str,
        out_column_list: str,
    ) -> pd.DataFrame:
        """
        Entry point for dynamic point selection.

        Args:
            X (pd.DataFrame): A pandas DataFrame with SMILES strings representing the chemical reactions.
            in_column_list (str): The name of the column containing the input SMILES strings.
            out_column_list (str): The name of the column to store the output SMILES strings.

        Returns:
            np.array: The filtered set of SMILES strings.
        """

        if self.verbose:
            print("------------------------------------------")
            print(
                "Performing dynamic filtering: batch size will be adjusted using previous statistics to ensure the desired number of points is reached:"
            )
        X = self.labeler(X=X, test_size=self.test_size, eval_size=self.eval_size)
        if self.shuffle:
            X = X.sample(frac=1).reset_index(drop=True)
        if not self.n_points:
            return self.process(
                X,
                in_column_list=in_column_list,
                out_column_list=out_column_list,
            )
        X_filtered = pd.DataFrame()
        while len(X_filtered) < self.n_points:
            batch = self.compute_batch(X, X_filtered)
            X_filtered = pd.concat(
                [
                    X_filtered,
                    self.process(
                        batch,
                        in_column_list=in_column_list,
                        out_column_list=out_column_list,
                    ),
                ],
                ignore_index=True,
            )
            if self.end_of_batch:
                return X_filtered
        if len(X_filtered) > self.n_points:
            X_filtered = X_filtered.iloc[: self.n_points]
        return X_filtered

    def compute_batch(self, X: pd.DataFrame, X_filtered: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the batch size for the next iteration of filtering, returns the batch of data to filter.

        Args:
            X (pd.DataFrame): A pandas DataFrame with SMILES strings representing the chemical reactions.
            X_filtered (pd.DataFrame): The filtered set of SMILES strings. This is used to compute the batch size.

        Returns:
            pd.DataFrame: The batch of data to filter.
        """

        if len(X_filtered):
            batch_size = max(
                50,
                int(
                    (self.index / len(X_filtered))
                    * (self.n_points - len(X_filtered))  # noqa W503
                ),
            )
            print(
                "Frequency of acceptance=",
                len(X_filtered) / max(self.index, 1),
            )

        else:
            batch_size = max(100, self.n_points)
        if self.verbose:
            print("Next batch_size=", batch_size)
        self.old_index = self.index
        self.index += batch_size

        if self.old_index + batch_size > len(X):
            self.end_of_batch = True
            batch_size = len(X) - self.old_index
            if batch_size == 0:
                if self.n_points < len(X):
                    print(
                        f"Warning: the filtering criteria are too strict. Only {len(X_filtered)} reactions were found."
                    )
                return pd.DataFrame()

        return X.iloc[self.old_index : (self.old_index + batch_size)]  # noqa E203
