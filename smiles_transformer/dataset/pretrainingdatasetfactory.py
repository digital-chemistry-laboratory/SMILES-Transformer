import pandas as pd
from datasets import Dataset, DatasetDict

from .basedatasetfactory import BaseDatasetFactory


class PretrainingDatasetFactory(BaseDatasetFactory):
    """Factory class for creating datasets specifically for pre-training purposes.

    This class extends DatasetFactory and is specialized in creating datasets suitable
    for pre-training machine learning models. It takes training and testing data,
    combines and transforms them into a Hugging Face
    DatasetDict object.

    Methods:
        create(X_train, X_test, y_train, y_test, model_mode): Creates and returns a DatasetDict object for pre-training.
    """

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
        """
        Creates and returns a DatasetDict object for pre-training.

        This method takes training and testing data, combines them with corresponding labels,
        and prepares a Hugging Face DatasetDict object which includes separate datasets for
        training and testing.

        Args:
            X_train (pd.DataFrame): Training data.
            y_train (pd.DataFrame): Training labels.
            X_eval (pd.DataFrame): Evaluation data.
            y_eval (pd.DataFrame): Evaluation labels.
            X_test (pd.DataFrame): Testing data.
            y_test (pd.DataFrame): Testing labels.

        Returns:
            DatasetDict: A Hugging Face DatasetDict object containing the training and testing datasets.

        Note:
            Only the "text" column is considered for dataset creation.
        """
        train_list = [X_train]
        eval_list = [X_eval]
        test_list = [X_test]
        if y_train is not None:
            train_list.append(y_train)
        if y_eval is not None:
            eval_list.append(y_eval)
        if y_test is not None:
            test_list.append(y_test)
        data_train = pd.concat(train_list, axis=1)

        data_eval = pd.concat(eval_list, axis=1)

        cols = ["text"]

        dataset_train = Dataset.from_pandas(
            data_train[cols],
            preserve_index=False,
            split="train",
            info=self.create_dataset_info(data_train[cols]),
        )
        dataset_eval = Dataset.from_pandas(
            data_eval[cols],
            preserve_index=False,
            split="eval",
            info=self.create_dataset_info(data_eval[cols]),
        )

        dataset = DatasetDict(
            {
                "train": dataset_train,
                "eval": dataset_eval,
            }
        )
        if X_test is not None:
            data_test = pd.concat(test_list, axis=1)
            dataset_test = Dataset.from_pandas(
                data_test[cols],
                preserve_index=False,
                split="test",
                info=self.create_dataset_info(data_test[cols]),
            )
            dataset["test"] = dataset_test
        dataset = self.encode_dataset(dataset)

        return dataset
