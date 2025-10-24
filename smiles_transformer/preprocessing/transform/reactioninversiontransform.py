import pandas as pd
from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
from tqdm import tqdm


class ReactionInversionTransform(TransformTemplate):
    """
    A transformation class for inverting the reactants and products in a reaction SMILES string.

    Attributes:
        not_smiles_error (str): A class attribute that stores the error message
            to be displayed if the input string does not contain '>>'.

    Properties:
        init_message (str): Returns a message indicating the nature of the
            transformation being applied. This is useful for logging and
            debugging purposes.
    """

    not_smiles_error = (
        "The given strings appear to not be reaction SMILES as they don't contain '>>'"
    )

    def step(self, batch):
        tqdm.pandas()
        return batch.progress_apply(self.reaction_inversion, axis=1)

    def reaction_inversion(self, row: pd.Series):
        """
        Inverts a reaction SMILES string.
        Args:
            row (pd.Series): A row of a DataFrame containing a reaction SMILES string.
        Returns:
            str: The inverted reaction SMILES string.
        """
        assert ">>" in row[self.in_column], self.not_smiles_error
        reaction = row[self.in_column]

        reactants, products = reaction.split(">>")
        row[self.out_column] = f"{products}>>{reactants}"
        return row

    @property
    def init_message(self) -> str:
        return "Inverting reactions: the reactants become the products and vice versa"
