from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np


class MorganDifferenceEncoder:
    """Reaction Morgan difference fingerprint: fp(products) - fp(reactants)."""

    def __init__(self, n_bits: int = 2048, radius: int = 3):
        self.n_bits = n_bits
        self.radius = radius
        self._gen = self._make_gen()

    def _make_gen(self):
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.n_bits
        )

    # --- make the encoder picklable for joblib/loky workers ---
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_gen", None)          # drop the unpicklable C++ generator
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._gen = self._make_gen()     # rebuild it in the worker process

    def _side_fp(self, side_smi: str) -> np.ndarray:
        fp = np.zeros(self.n_bits, dtype=np.int32)
        if not side_smi:
            return fp
        for s in side_smi.split("."):
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {s!r}")
            fp += self._gen.GetCountFingerprintAsNumPy(mol).astype(np.int32)
        return fp

    def _encode_one(self, rxn_smiles: str) -> np.ndarray:
        parts = rxn_smiles.split(">")
        if len(parts) == 2:
            reactants_smi, products_smi = parts
        elif len(parts) == 3:
            reactants_smi, _agents, products_smi = parts
        else:
            raise ValueError(f"Invalid reaction SMILES: {rxn_smiles!r}")
        return self._side_fp(products_smi) - self._side_fp(reactants_smi)

    def encode(self, smiles):
        if isinstance(smiles, str):
            smiles = [smiles]
        return [self._encode_one(s) for s in smiles]