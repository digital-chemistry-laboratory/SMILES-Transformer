import pandas as pd

from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
from tqdm import tqdm


# TODO: add TESTS
class CGRtoStrTransform(TransformTemplate):
    """ """

    def step(self, batch):
        """
        Transform the CGR strings in into SILES/CGR strings.
        if wanted, specify n_augmentations in the class init to augment the SMILES/CGR strings.
        Augments only the training set.
        Args:
            batch (pd.DataFrame): The batch of data to augment. It should contain CGR containers in its main column.

        Returns:
            pd.DataFrame: The augmented batch of SMILES/CGR strings.
        """
        augmented_batch = []
        for _, row in tqdm(batch.iterrows()):
            cgr = row[self.in_column]
            augment = True
            if "split" in row.index:
                if row["split"] != "train":
                    augment = False
            augmented_cgrs = self.augment_cgr(cgr, augment)
            for augmented_cgr in augmented_cgrs:
                new_row = row.copy()
                new_row[self.out_column] = augmented_cgr
                augmented_batch.append(new_row)
        return pd.DataFrame(augmented_batch)

    def augment_cgr(self, cgr, augment=True):
        """
        Augment a CGR into random SMILES/CGR.

        Args:
            cgr (str): A CGR container.

        Returns:
            str: an array with the augmented SMILES/CGR strings.
        """
        cgrs = []
        if self.kwargs.get("n_augmentations", None) and (augment == True):
            tostr = lambda x: f"{x:r}"  # Noqa E731
        else:
            return [str(cgr)]
        for _ in range(self.kwargs["n_augmentations"]):
            cgrs.append(tostr(cgr))
        return cgrs

    @property
    def init_message(self) -> str:
        if self.kwargs.get("n_augmentations", None):
            return f"Augmenting SMILES/CGR: one CGR will become {self.kwargs['n_augmentations']} randomized SMILES/CGR"
        else:
            return "Convert CGR to SMILES/CGR without augmentation"
