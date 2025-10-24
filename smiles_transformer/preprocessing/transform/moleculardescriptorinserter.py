from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
import pandas as pd
from tqdm import tqdm


class MolecularDescriptorInserter(TransformTemplate):
    """
    Incoorporates given molecular descriptors into SMILES.
    """

    def _init_descriptorinserter_params(self):
        """Setup needed parameters."""

        self.not_list_error = "molecular_descriptors is not a list. It has to be a list, even with a single selected descriptor"
        self.format_specifiers_type_error = "format_specifiers must be a list of strings that specify the format of the inserted bins."
        self.format_specifiers_len_error = "format_specifiers must be a list of length 5. Even if the some elements are not to be displayed, they must be contained as empty strings '' "

        self.format_specifiers = ["[", "]", ":"]
        self.molecular_descriptors = self.kwargs.get("molecular_descriptors", [])
        self.numbered_smiles_column = self.in_column
        self.inserted_smiles_column = self.out_column

        assert isinstance(self.molecular_descriptors, list), self.not_list_error
        assert isinstance(
            self.format_specifiers, list
        ), self.format_specifiers_type_error

    def step(self, batch):
        """
        Add atomic and molecular descriptors based on what was chosen.
        """
        self._init_descriptorinserter_params()
        tqdm.pandas()
        # Add in the molecular descriptors descriptors first, then use these added atomics tp append the molecular ones
        batch[self.inserted_smiles_column] = batch.progress_apply(
            self.append_molecular_descriptors,
            axis=1,
            molecular_descriptors=self.molecular_descriptors,
            numbered_smiles_column=self.numbered_smiles_column,
            format_specifiers=self.format_specifiers,
        )

        return batch

    def append_molecular_descriptors(
        self,
        row: pd.Series,
        molecular_descriptors: list,
        numbered_smiles_column: str = "smiles",
        format_specifiers: list = ["[MOLDESC]", "[", "]", ":", "[ENDESCMOL]"],
    ):
        """
        Appends molecular descriptors to the end of a SMILES string using custom formatting.

        Args:
        - row (Series): A pandas Series or similar containing the SMILES string and descriptors.
        - molecular_descriptors (list): A list of descriptor names present as keys in the row.
        - format_specifiers (list): List of five elements specifying the symbols used in binning.

        Returns:
        - str: The SMILES string with appended molecular descriptors.
        """
        smiles = row[numbered_smiles_column]

        # Validate format specifiers

        pt_start, pt_end, middle = format_specifiers

        # Compile the descriptor string using the provided format
        descriptor_str = "".join(
            f"{pt_start}{name}{middle}{row[name]}{pt_end}"
            for name in molecular_descriptors
            if name in row
        )

        # Append the formatted descriptor string to the SMILES and return it
        return f"{descriptor_str}%%%%{smiles}"

    @property
    def init_message(self) -> str:
        return "Appending Molecular descriptors to the SMILES either containing atomic descriptors or none at all."
