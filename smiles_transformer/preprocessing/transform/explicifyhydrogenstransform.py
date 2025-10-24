from rdkit import Chem
from rdkit.Chem import AllChem
from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
import re


# TODO: add TESTS
class ExplicifyHydrogensTransform(TransformTemplate):
    """
    Forces hydrogens to be explicit.
    """

    pattern = re.compile("\[[A-Z][a-z]?@?:\d*\]")

    def step(self, batch):
        self.is_smiles_reaction(batch[self.in_column].iloc[0])
        if self.kwargs["mapping"] is not None:
            mapping_removal = "remove" in self.kwargs["mapping"]
        else:
            mapping_removal = False

        assert (
            not self.pattern.match(batch[self.in_column].iloc[0]) or mapping_removal
        ), "Error: Explicit hydrogens for reactions only work for reactions that are not atom-mapped. Set mapping to 'remove'"

        try:

            batch[self.out_column] = batch[self.in_column].apply(
                self.explicify_hydrogens_reactions
            )

        except:
            batch[self.out_column] = batch[self.in_column].apply(
                self.explicify_hydrogens_molecules
            )
        else:
            print("ExplicifyHydrogens skipped, reaction or molecule not recognized")

        return batch

    def is_smiles_reaction(self, rxn_smiles):
        # Parse the reaction from SMILES; raises an error if the SMILES is invalid.
        assert ">>" in rxn_smiles, "This is not a reaction SMILES string"
        rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        if rxn is None:
            raise ValueError("Invalid reaction SMILES")

        # Generator expressions are used so that evaluation stops immediately when an unmapped atom (map number 0) is found.
        all_reactants_mapped = all(
            atom.GetAtomMapNum()
            for mol in rxn.GetReactants()
            for atom in mol.GetAtoms()
        )
        all_products_mapped = all(
            atom.GetAtomMapNum() for mol in rxn.GetProducts() for atom in mol.GetAtoms()
        )

        return all_reactants_mapped and all_products_mapped

    def explicify_hydrogens_reactions(self, rxn_smiles):
        reactants, products = rxn_smiles.split(">>")
        explicit_reactants = []
        explicit_products = []
        for reactant in reactants.split("."):
            explicit_reactants.append(self.explicify_hydrogens_molecules(reactant))
        for product in products.split("."):
            explicit_products.append(self.explicify_hydrogens_molecules(product))
        explicit_reaction = (
            ".".join(explicit_reactants) + ">>" + ".".join(explicit_products)
        )
        return explicit_reaction

    def explicify_hydrogens_molecules(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        # print(f"smiles: {smiles}")
        # print(f"extended H: {Chem.MolToSmiles(Chem.AddHs(mol))}")
        return Chem.MolToSmiles(Chem.AddHs(mol))

    @property
    def init_message(self) -> str:
        return "Explicify Hydrogens"
