import re

from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
from tqdm import tqdm


# TODO: add TESTS
class CGRBondChangeCompute(TransformTemplate):
    """
    Compute new column n_bond_change with the number of bond changes in the reaction.
    """

    def step(self, batch):
        tqdm.pandas()
        regex = re.compile(r"\[.>.\]")
        batch.loc[:, self.out_column] = (
            batch.loc[:, self.in_column]
            .progress_apply(lambda row: len(regex.findall(row)))
            .values
        )

        return batch

    @property
    def init_message(self) -> str:
        return "computing CGR bond/charge changes"
