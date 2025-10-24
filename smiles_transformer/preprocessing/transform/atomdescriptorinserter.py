from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
import re
#from tqdm import tqdm
import swifter


class AtomDescriptorInserter(TransformTemplate):
    """
    Incoorporates given atomic descriptors into SMILES.
    Do not use after molecular descriptor inserter!
    """

    def _setup_descriptorinserter(self):
        """
        Sets up parameters needed to work with the MPP datasets.

        This method initializes attributes such as format specifiers,
        selected atomic descriptors, and the column mappings for the input
        and output SMILES strings. It also handles configurations for
        empty or null descriptors.
        """

        self.atomic_descriptors = self.kwargs.get("atomic_descriptors")
        # Hanlde all and none descriptors from config
        if self.atomic_descriptors in [None, False, "null"]:
            self.atomic_descriptors = []

        return

    def step(self, batch):
        """
        Add atomic descriptors to batch.
        Args:
            batch (pd.DataFrame): The batch of SMILES and descriptors.

        Returns:
            pd.DataFrame: The inserted SMILES.
        """
        self._setup_descriptorinserter()
        #tqdm.pandas()

        # Add in the atomic descriptors first, then use these added atomics tp append the molecular ones
        batch[self.out_column] = batch.swifter.progress_bar(enable=True).apply(
            self.add_atomic_descriptors,
            axis=1,
            atomic_descriptors=self.atomic_descriptors,
            mapped_smiles_column=self.in_column,
        )

        return batch

    def add_atomic_descriptors(
        self,
        row,
        mapped_smiles_column,
        atomic_descriptors,
    ):
        """
        Appends atomic descriptors as strings behind each atom block in the SMILES string.

        Args:
        - row (pd.Series): Data row containing the mapped SMILES string and descriptors.
        - mapped_smiles_column (str): Column name containing the SMILES string with numbered atoms.
        - atomic_descriptors (list of str): List of descriptor names used as keys in row.
        - format_specifiers (list of str): Format specifiers for enclosing each descriptor.

        Returns:
        - str: Modified SMILES string with descriptors appended to each atom block.
        """
        smiles = row[mapped_smiles_column]
        pattern = re.compile(r"(\[[^\]]+?:\d+\])")
        matches = list(re.finditer(pattern, smiles))

        pt_start, pt_end, middle = ["[", "]", ":"]
        inserted_smiles = smiles
        # Reverse to prevent index shifting during replacement
        for match in reversed(matches):
            atom_block = match.group(1)
            atom_index = (
                int(re.search(r":(\d+)\]", atom_block).group(1)) - 1
            )  # assuming 1-based index

            descriptor_strs = []
            for name in atomic_descriptors:
                descriptor_value = row[name][atom_index]
                descriptor_strs.append(
                    f"{pt_start}{name}{middle}{descriptor_value}{pt_end}"
                )

            new_atom_block = atom_block + "".join(descriptor_strs)
            inserted_smiles = inserted_smiles.replace(
                atom_block, self.remove_mapping(new_atom_block)
            )
        return inserted_smiles

    def remove_mapping(self, smiles):
        """
        Removes atom mapping from the SMILES string.

        Args:
        - smiles (str): SMILES string with atom mapping.

        Returns:
        - str: SMILES string without atom mapping.
        """
        pattern_to_remove = re.compile(r"\[[a-zA-Z]{1,2}:\d{1,3}\]")
        pattern_to_place = re.compile(r"[a-zA-Z]{1,2}")
        return re.sub(
            pattern_to_remove, re.search(pattern_to_place, smiles).group(0), smiles
        )

    @property
    def init_message(self) -> str:
        return "Inserting atomic descriptors into smiles."
