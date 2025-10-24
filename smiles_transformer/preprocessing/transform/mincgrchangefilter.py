import re

from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)


# TODO: add TESTS
class MinCGRChangeFilter(TransformTemplate):
    """
    Filter out datapoints with less than n_min_bond_changes CGR changes.
    This class needs a kwargs argument n_min_bond_changes.
    """

    no_n_min_bond_changes_error = (
        "n_min_bond_changes not in given. Please provide n_min_bond_changes in kwargs."
    )

    def step(self, batch):
        """
        The step method. This method is used to implement the processing logic.
        This method must be implemented by the subclass.

        Args:
            batch (pd.DataFrame): The batch of data to process.

        Returns:
            pd.DataFrame: The processed batch of data.
        """
        assert "n_min_bond_changes" in self.kwargs, self.no_n_min_bond_changes_error
        regex = re.compile(r"\[.>.\]")

        if "n_CGR_changes" not in batch.columns:
            batch.loc[:, self.out_column] = (
                batch.loc[:, self.in_column]
                .apply(lambda row: len(regex.findall(row)))
                .values
            )

        return batch[(batch[self.out_column] >= self.kwargs.get("n_min_bond_changes"))]

    @property
    def init_message(self) -> str:
        return f"filtering out datapoints with less than {self.kwargs['n_min_bond_changes']} CGR changes"
