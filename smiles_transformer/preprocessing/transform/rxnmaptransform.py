from rxnmapper import BatchedMapper

from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)
from tqdm import tqdm


class RXNMapTransform(TransformTemplate):
    """
    remap reactions using rxnmapper.
    takes self.in_column and creates self.out_column with the remapped reactions.
    """

    def step(self, batch):
        rxn_mapper = BatchedMapper(batch_size=64)

        remapped_reactions = list(
            rxn_mapper.map_reactions_with_info(batch[self.in_column].to_numpy())
        )
        confidence = []
        remapped_rxn = []
        for reaction, input_str in tqdm(zip(remapped_reactions, batch[self.in_column])):
            if "mapped_rxn" in reaction.keys():
                remapped_rxn.append(reaction["mapped_rxn"])
                confidence.append(reaction["confidence"])
            else:
                remapped_rxn.append("")
                confidence.append("")
                print(f"Error in mapping reaction: {input_str}")

        batch["confidence"] = confidence
        batch[self.out_column] = remapped_rxn
        if self.verbose:
            print("Remapping done!")
        return batch

    @property
    def init_message(self) -> str:
        return "Mapping reactions..."
