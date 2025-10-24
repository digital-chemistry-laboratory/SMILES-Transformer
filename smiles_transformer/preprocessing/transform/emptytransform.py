import pandas as pd

from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)


# TODO: add TESTS
class EmptyTransform(TransformTemplate):
    """
    Does nothing. .
    """

    def step(self, batch):

        return batch

    @property
    def init_message(self) -> str:
        return "Empty transform"
