import re

from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)


# TODO: add TESTS
class MaxCGRChangeFilter(TransformTemplate):
    """
    Filter out datapoints with more than n_max_bond_changes CGR changes.
    This class needs a kwargs argument n_max_bond_changes.
    """

    no_n_max_bond_changes_error = "n_max_bond_changes not in given. Please provide n_max_bond_changes in kwargs of the class init or the step function."

    def step(self, batch):
        assert "n_max_bond_changes" in self.kwargs, self.no_n_max_bond_changes_error

        regex = re.compile(r"\[.>.\]")
        if "n_CGR_changes" not in batch.columns:
            batch.loc[:, self.out_column] = (
                batch.loc[:, self.in_column]
                .apply(lambda row: len(regex.findall(row)))
                .values
            )

        return batch[(batch[self.out_column] <= self.kwargs.get("n_max_bond_changes"))]

    @property
    def init_message(self) -> str:
        return f"filtering out datapoints with more than {self.kwargs['n_max_bond_changes']} CGR changes"
