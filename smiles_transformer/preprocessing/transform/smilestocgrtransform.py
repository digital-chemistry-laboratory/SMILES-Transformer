from CGRtools.files import SMILESRead

from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
from tqdm import tqdm


# TODO: add TESTS
class SMILEStoCGRTransform(TransformTemplate):
    """
    Convert from vanilla atom-mapped smiles reactions to CGR using CGRtools.
    Takes self.column_name["in"] and creates self.column_name["out"] with the remapped reactions.
    """

    not_smiles_error = (
        "The given strings appear to not be reaction SMILES as they don't contain '>>'"
    )
    parser = SMILESRead.create_parser(ignore=True)
    num_failed_conversions = 0

    def auto_convert(self, smiles: str, explicify_hydrogens=True) -> str:
        """
        Converts an atom-mapped SMILES string to a CGR string if it is a vanilla atom-mapped SMILES.
        Implicifies hydrogens.

        Args:
            smiles (str): A SMILES string.
            explicify_hydrogens (bool, optional): Whether to implicify hydrogens. Defaults to True.

        Returns:
            CGR container: The CGR string.
        """  # noqa 501

        smiles = str(smiles)

        try:
            m = self.parser(smiles)
            if explicify_hydrogens:
                m.explicify_hydrogens()

            try:
                return m.compose()
            except TypeError:
                return m
        except:  # noqa 722
            if self.verbose:
                print(f"Could not convert SMILES to CGR. input: {smiles}")
            self.num_failed_conversions += 1
            return ""

    def step(self, batch):
        """
        Convert the SMILES strings in the batch to CGR graphs.
        """
        tqdm.pandas()
        batch[self.out_column] = (
            batch[self.in_column]
            .progress_apply(
                lambda x: self.auto_convert(
                    x,
                    self.kwargs.get("explicify_hydrogens", True),
                )
            )
            .values
        )
        return batch

    @property
    def init_message(self) -> str:
        return "Converting SMILES into CGR containers to be manipulated and used for further processing."
