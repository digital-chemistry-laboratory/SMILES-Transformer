from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)


# TODO: add TESTS
class MappingConfidenceFilter(TransformTemplate):
    """
    Filter out datapoints with less than the remapping confidence.
    This class should be used AFTER the RXNMapTransform.
    When initializing this class, remap_confidence_threshold should be provided in the kwargs.
    If the confidence column is not named 'confidence', remap_confidence_column_name should be provided in the kwargs.
    """

    no_confidence_column_error = "The 'confidence' column could not be found, and a valid column was not provided. Please provide this column, or use the RXNMapTransform before this class."
    no_confidence_threshold_error = "remap_confidence_threshold not in given. Please provide remap_confidence_threshold in kwargs of the class init."

    def step(self, batch):
        """
        The step method. This method is used to implement the processing logic.
        This method must be implemented by the subclass.

        Args:
            batch (pd.DataFrame): The batch of data to process.

        Returns:
            pd.DataFrame: The processed batch of data.
        """
        assert (
            "remap_confidence_threshold" in self.kwargs
        ), self.no_confidence_threshold_error
        assert ("confidence" in batch.columns) or (
            "remap_confidence_column_name" in self.kwargs
        ), self.no_confidence_column_error

        return batch[
            (
                batch[self.kwargs.get("remap_confidence_column_name", "confidence")]
                >= self.kwargs.get("remap_confidence_threshold")
            )
        ]

    @property
    def init_message(self) -> str:
        return "filtering out datapoints with low mapping confidence"
