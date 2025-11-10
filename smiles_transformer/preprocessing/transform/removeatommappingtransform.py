from rdkit import Chem

from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)

from tqdm import tqdm


# TODO: add TESTS
class RemoveAtomMappingTransform(TransformTemplate):
    """
    Convert from atom-mapped smiles reactions to vanilla non-atom-mapped smiles reactions.
    Additional kwargs:
        kekulesmiles (bool): Whether to use Kekule SMILES or not. Defaults to True.
    """

    not_smiles_error = (
        "The given strings appear to not be reaction SMILES as they don't contain '>>'"
    )

    def step(self, batch):
        """
        Converts atom-mapped SMILES strings to vanilla SMILES strings.

        Args:
            batch (pd.DataFrame): The batch of data to filter.

        Returns:
            pd.DataFrame: The filtered batch of data.
        """
        tqdm.pandas()
        batch.loc[:, self.out_column] = batch[self.in_column].progress_apply(
            self.remove_mapping
        )
        return batch

    def remove_mapping(self, smiles: str) -> str:
        """
        Converts an atom-mapped SMILES string to a SMILES/CGR string if it is a vanilla atom-mapped SMILES.

        Args:
            smi (str): A SMILES string.

        Returns:
            str: The SMILES/CGR string.
        """  # noqa 501

        smiles = str(smiles)
        if (">>" not in smiles) and self.verbose:
            print(smiles + " :\n", self.not_smiles_error)

        reactants, products = smiles.split(">>")

        reactants_mol = Chem.MolFromSmiles(reactants)
        products_mol = Chem.MolFromSmiles(products)

        def clear_atom_mapping(mol, smi):
            if mol is None and self.verbose:
                print(f"smi = {smi}")
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
            return mol

        reactants_mol = clear_atom_mapping(reactants_mol, reactants)
        products_mol = clear_atom_mapping(products_mol, products)

        reactants_smiles = Chem.MolToSmiles(
            reactants_mol,
            canonical=False,
            kekuleSmiles=self.kwargs.get("kekulesmiles", True),
        )
        products_smiles = Chem.MolToSmiles(
            products_mol,
            canonical=False,
            kekuleSmiles=self.kwargs.get("kekulesmiles", True),
        )

        non_mapped_smiles = f"{reactants_smiles}>>{products_smiles}"
        return non_mapped_smiles

    @property
    def init_message(self) -> str:
        return (
            "Removing atom mapping\n"
            + ("NOT u" if not self.kwargs.get("kekulesmiles", True) else "U")
            + "sing Kekule SMILES"
        )
