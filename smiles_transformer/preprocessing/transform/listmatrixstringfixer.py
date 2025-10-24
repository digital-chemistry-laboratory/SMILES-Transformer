from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
import ast
from tqdm import tqdm
import swifter


class ListMatrixStringFixer(TransformTemplate):
    """
    Converts string-represented lists and matrices in a data batch into actual lists and matrices.
    """

    def step(self, batch):
        """
        Transforms strings to list/matrix types for specified batch descriptors.

        Args:
            batch (pd.DataFrame): Data batch with string representations to be converted.

        Modifies `batch` in-place.
        """

        self.descriptors_to_bin = self.kwargs.get("descriptors_to_bin")

        for descriptor in tqdm(
            self.descriptors_to_bin, desc="descriptor columns processed"
        ):
            if isinstance(batch[descriptor].iloc[0], str):
                batch[descriptor] = batch[descriptor].swifter.apply(ast.literal_eval)
        return batch

    @property
    def init_message(self) -> str:
        """Returns initialization message for the transformation."""

        return f"""Converting string representations of descriptor lists to literals"""
