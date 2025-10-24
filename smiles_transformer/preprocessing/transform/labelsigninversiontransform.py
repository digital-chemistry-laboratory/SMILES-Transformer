from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
from tqdm import tqdm


class LabelSignInversionTransform(TransformTemplate):
    """
    A transformation class for inverting the sign of labels in a batch of data.

    NOTE: This transformation does not use in and out columns! Instead, it takes the label_column_name from the kwargs.

    Attributes:
        no_label_col_error (str): A class attribute that stores the error message
            to be displayed if the `label_column_name` is not provided.

    Properties:
        init_message (str): Returns a message indicating the nature of the
            transformation being applied. This is useful for logging and
            debugging purposes.
    """

    no_label_col_error = "label_column_name must be provided"

    def step(self, batch):
        """
        Applies the label sign inversion transformation to the specified column in the batch.

        This method retrieves the name of the label column from the instance's `kwargs`
        attribute, asserts its presence, and then applies a function to invert the sign
        of every value in that column of the input batch.

        Args:
            batch (pd.DataFrame): A batch of data, expected to be a dictionary where keys are
                column names and values are lists or arrays of data.

        Raises:
            AssertionError: If the `label_column_name` is not found in the `kwargs`
                provided to the instance.

        Returns:
            pd.DataFrame: The batch of data with the sign of the label values inverted.
        """
        label_column_name = self.kwargs.get("label_column_name")
        assert label_column_name is not None, self.no_label_col_error
        tqdm.pandas()

        batch[label_column_name] = batch[label_column_name].progress_apply(lambda x: -x)
        return batch

    @property
    def init_message(self) -> str:
        return (
            "inverting label signs\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
            "This transformation inverts the sign of the labels in the dataset. "
            "This is useful for tasks where the sign of the label CAN be inverted, "
            "such as reaction enthalpy. Be careful when using this transformation, "
            "as it may lead to unexpected results if the sign of the labels"
        )
