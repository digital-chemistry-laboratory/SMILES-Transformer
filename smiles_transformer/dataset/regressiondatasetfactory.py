import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

from .basedatasetfactory import BaseDatasetFactory


class RegressionDatasetFactory(BaseDatasetFactory):
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
        self.target_median = np.median(y_train)
        if scale_target:
            self.std = np.std(dataset["train"]["label"])
            self.mean = np.mean(dataset["train"]["label"])

            with open(
                os.path.join(self.output_dir, "scaler.json"),
                "w",
            ) as f:
                json.dump({"mean": self.mean, "std": self.std}, f)
            for split in dataset:
                dataset[split] = dataset[split].map(
                    lambda x: {"label": ((x - self.mean) / self.std)},
                    input_columns=["label"],
                    batched=True,
                )

        dataset = self.encode_dataset(dataset)

        return dataset
