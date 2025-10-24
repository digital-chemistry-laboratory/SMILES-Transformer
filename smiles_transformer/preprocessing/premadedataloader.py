import os

import numpy as np
import pandas as pd


from smiles_transformer.preprocessing.basedataloader import BaseDataLoader
from smiles_transformer.utils.path_finder import path_finder


class PremadeDataLoader(BaseDataLoader):
    """
    Dataloader class for loading premade datasets supporting cross-validation.

    Attributes:
        test_size_not_str (str): Error message displayed if `test_size` is not a string.
    """

    test_size_not_str = (
        "test_size must be a string. This dataloader is only used for "
        "crossvalidation with premade datasets, are you sure that this is what you are trying to do?"
    )

    def load_dataset(
        self,
        in_column_list: str,
        out_column_list: str,
        separator: str = "\t",
    ) -> np.ndarray:
        """
        Loads a premade dataset for cross-validation from a specified directory.

        Args:
            in_column_list (str): Names of the input columns.
            out_column_list (str): Names of the output columns.
            separator (str, optional): Delimiter used in the CSV files. Defaults to "\t".

        Returns:
            np.ndarray: Processed dataset ready for use.
        """
        base_directory = path_finder(self.path, self.dataset_name.replace(".csv", ""))
        assert type(self.test_size) is str, self.test_size_not_str
        fold_number = int(self.test_size.split("/")[0])
        directory = os.path.join(base_directory, f"fold_{fold_number-1}")

        df_train = self.init_pandas(
            os.path.join(directory, "aam_train.csv"), separator, n_points=self.n_points
        )
        df_train["split"] = "train"

        df_test = self.init_pandas(
            os.path.join(directory, "aam_test.csv"), separator, n_points=None
        )
        df_test["split"] = "test"

        df_eval = self.init_pandas(
            os.path.join(directory, "aam_val.csv"), separator, n_points=None
        )
        df_eval["split"] = "eval"

        if self.verbose:
            print(f"loaded data for fold {fold_number}")
        df_full = pd.concat([df_train, df_test, df_eval], ignore_index=True)
        df_full["original_input"] = df_full[self.original_column_name]
        return self.process(
            df_full,
            in_column_list=in_column_list,
            out_column_list=out_column_list,
        )

    def init_pandas(self, path, separator, n_points=None):

        if n_points:
            if n_points < 1:
                total_len = len(pd.read_csv(path, sep=separator, on_bad_lines="skip"))
                n_points = int(n_points * total_len)
            return pd.read_csv(
                path,
                sep=separator,
                on_bad_lines="skip",
                nrows=n_points,
            )
        return pd.read_csv(
            path,
            sep=separator,
            on_bad_lines="skip",
        )
