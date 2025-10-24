try:
    from smiles_transformer.preprocessing.transform.transformtemplate import (
        TransformTemplate,
    )
except ModuleNotFoundError:
    from transformtemplate import TransformTemplate

from rdkit import Chem
from tqdm import tqdm


# TODO: add TESTS
class CorrectNitroTransform(TransformTemplate):
    """
    Perform correction on SMILES for nitro groups and radicals
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
        tqdm.pandas()
        batch[self.out_column] = batch[self.in_column].progress_apply(
            self.fix_reaction,
        )
        return batch

    def fix_reaction(self, reaction):
        if ">>" not in reaction:
            raise ValueError(self.not_smiles_error)

        reactants, products = reaction.split(">>")
        reactants = reactants.split(".")
        products = products.split(".")
        fixed_reactants = []
        fixed_products = []
        fixed_reactants_str = []
        fixed_products_str = []
        for i, reactant in enumerate(reactants):
            fixed_reactants.append(self.fix_mol(Chem.MolFromSmiles(reactant)))
            fixed_reactants_str.append(
                Chem.MolToSmiles(fixed_reactants[i]).split(".")[0]
            )
        for i, product in enumerate(products):
            fixed_products.append(self.fix_mol(Chem.MolFromSmiles(product)))
            fixed_products_str.append(Chem.MolToSmiles(fixed_products[i]).split(".")[0])

        return f"{'.'.join(fixed_reactants_str)}>>{'.'.join(fixed_products_str)}"

    def fix_mol(self, mol):
        rw_mol = Chem.RWMol(mol)
        for atom in rw_mol.GetAtoms():
            if atom.GetNumRadicalElectrons() >= 1:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetNumRadicalElectrons() >= 1:
                        bond = rw_mol.GetBondBetweenAtoms(
                            atom.GetIdx(), neighbor.GetIdx()
                        )
                        rw_mol.RemoveBond(atom.GetIdx(), neighbor.GetIdx())
                        rw_mol.AddBond(
                            atom.GetIdx(),
                            neighbor.GetIdx(),
                            Chem.BondType.values[int(bond.GetBondType()) + 1],
                        )
                        atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 1)
                        neighbor.SetNumRadicalElectrons(
                            neighbor.GetNumRadicalElectrons() - 1
                        )
                        break
        mol = Chem.Mol(rw_mol)
        Chem.SanitizeMol(mol)
        return mol

    @property
    def init_message(self) -> str:
        return "Fixing nitro groups in reaction SMILES."
