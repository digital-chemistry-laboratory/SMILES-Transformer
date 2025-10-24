import json
import os

from collections import Counter
import pandas as pd
from datasets import Dataset, DatasetDict

from .basedatasetfactory import BaseDatasetFactory
from sklearn.preprocessing import LabelEncoder


class ClassificationDatasetFactory(BaseDatasetFactory):
    def create(
        self,
        X_train,
        y_train,
        X_eval: pd.DataFrame = None,
        y_eval: pd.DataFrame = None,
        X_test=None,
        y_test=None,
        scale_target: bool = False,
    ):
        self.label_counter = Counter(y_train)

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(y_train)

        y_train = pd.Series(self.label_encoder.transform(y_train), name="label")
        y_eval = pd.Series(self.label_encoder.transform(y_eval), name="label")
        y_test = pd.Series(self.label_encoder.transform(y_test), name="label")

        assert self.output_dir is not None, self.no_output_dir_error
        data_train = pd.concat(
            [X_train.reset_index(drop=True), y_train.reset_index(drop=True)],
            axis=1,
        )

        data_eval = pd.concat(
            [X_eval.reset_index(drop=True), y_eval.reset_index(drop=True)],
            axis=1,
        )
        if X_test is not None:
            data_test = pd.concat(
                [X_test.reset_index(drop=True), y_test.reset_index(drop=True)],
                axis=1,
            )

        cols = ["text", "label"]
        cols = cols + self.additional_features

        dataset_dict = {
            "train": Dataset.from_pandas(
                data_train[cols],
                preserve_index=False,
                info=self.create_dataset_info(data_train[cols]),
            )
        }
        dataset_dict["eval"] = Dataset.from_pandas(
            data_eval[cols],
            preserve_index=False,
            info=self.create_dataset_info(data_eval[cols]),
        )

        if X_test is not None:
            dataset_dict["test"] = Dataset.from_pandas(
                data_test[cols],
                preserve_index=False,
                info=self.create_dataset_info(data_test[cols]),
            )
        dataset = DatasetDict(dataset_dict)

        dataset = self.encode_dataset(dataset)

        return dataset
