import pandas as pd

from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
from tqdm import tqdm
from sr_smiles import rxn_to_sr


class SMILEStoSRSMILESTransform(TransformTemplate):

    def step(self, batch):
        """
        Transform SMILES strings to sr-SMILES strings.
        augmentations should be made beforehand with the reaction SMILES
        Args:
            batch (pd.DataFrame): The batch of data to augment. It should contain SMILES in its main column.

        Returns:
            pd.DataFrame: The augmented batch of sr-SMILES strings.
        """

        sr_smiles_settings = {
            list(item.keys())[0]: item[list(item.keys())[0]]
            for item in self.kwargs["sr_smiles_settings"]
        }

        new_batch = []
        for _, row in tqdm(batch.iterrows()):
            smiles = row[self.in_column]
            row[self.out_column] = rxn_to_sr(smiles, **sr_smiles_settings)
            new_batch.append(row)
        return pd.DataFrame(new_batch)

    @property
    def init_message(self) -> str:
        return "Convert SMILES to sr-SMILES"
