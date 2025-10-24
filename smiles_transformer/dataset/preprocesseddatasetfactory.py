import json

import pandas as pd
from datasets import DatasetDict

from smiles_transformer.utils.path_finder import path_finder

from .basedatasetfactory import BaseDatasetFactory


class PreprocessedDatasetFactory(BaseDatasetFactory):
    """Factory class for loading an already preprocessed and tokenized dataset."""

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
        path = self.load_dataset_path
        if self.verbose:
            print("-----------------------------------------")
            print(f"loading preprocessed dataset from {path}")

        dataset = DatasetDict.load_from_disk(path)

        if self.verbose:
            for split in dataset:
                print(f"-> {split} dataset length: {len(dataset[split])}")
            print(f"-> Dataset config: {dataset['train'].info.description}")
            print("-----------------------------------------")
        return dataset
