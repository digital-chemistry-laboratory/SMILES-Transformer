try:
    from smiles_transformer.preprocessing.transform.transformtemplate import (
        TransformTemplate,
    )
except ModuleNotFoundError:
    from transformtemplate import TransformTemplate
from tqdm import tqdm


import pandas as pd
from rdkit import Chem


# TODO: add TESTS
class AugmentSMILESTransform(TransformTemplate):
    """
    Perform SMILES augmentation on the dataset.
    Needs to be inittialized with a number of smiles to generate for each input smiles.
    """

    not_smiles_error = (
        "The given strings appear to not be reaction SMILES as they don't contain '>>'"
    )
    no_n_augmentation = (
        "n_augmentations not given. Please provide n_augmentations in the class init."
    )

    def step(self, batch):
        """
        Augment the SMILES strings in the batch.
        Args:
            batch (pd.DataFrame): The batch of data to augment.

        Returns:
            pd.DataFrame: The augmented batch of data.
        """
        assert "n_augmentations" in self.kwargs, self.no_n_augmentation
        augmented_batch = []
        for _, row in tqdm(batch.iterrows()):
            if "split" in row.index:
                if row["split"] != "train":
                    augmented_batch.append(row)
                    continue
            smiles_reaction = row[self.in_column]
            augmented_reactions = self.augment_reaction(smiles_reaction)
            for augmented_reaction in augmented_reactions:
                new_row = row.copy()
                new_row[self.out_column] = augmented_reaction
                augmented_batch.append(new_row)
        return pd.DataFrame(augmented_batch)

    def augment_reaction(self, smiles: str) -> str:
        """
        Augment a SMILES reaction.

        Args:
            smi (str): A SMILES reaction string.

        Returns:
            str: The augmented SMILES reaction string.
        """  # noqa 501
        if (">>" not in smiles) and self.verbose:
            print(smiles + " :\n", self.not_smiles_error)

        reactants, products = smiles.split(">>")
        augmented_reactants = self.augment_mol(reactants)
        augmented_products = self.augment_mol(products)
        augmented_reaction = []
        for reactant, product in zip(augmented_reactants, augmented_products):
            augmented_reaction.append(f"{reactant}>>{product}")
        return augmented_reaction

    def augment_mol(self, smi):
        """
        Augment a SMILES molecule.

        Args:
            smi (str): A SMILES molecule string.

        Returns:
            str: an array with the augmented SMILES strings.
        """
        mols = []
        mol = Chem.MolFromSmiles(smi)
        for _ in range(self.kwargs["n_augmentations"]):
            mols.append(Chem.MolToSmiles(mol, doRandom=True, canonical=False))
        return mols

    @property
    def init_message(self) -> str:
        return f"Augmenting SMILES: one SMILES will become {self.kwargs['n_augmentations']} randomized SMILES"
