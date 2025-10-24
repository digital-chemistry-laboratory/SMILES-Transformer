from transformers import BertTokenizer
from abc import ABC, abstractmethod
import os


class BaseTokenizerTemplate(BertTokenizer, ABC):
    """
    THIS CLASS IS AN ABSTRACT CLASS. DO NOT INSTANTIATE IT.
    The inheriting class must define a self.pattern attribute.

    Constructs a SmilesBertTokenizer. Uses the tokenizer defined in the Tokenizer class.
    Taken from https://github.com/rxn4chemistry/rxnfp/blob/master/nbs/01_tokenization.ipynb
    Adapted from https://github.com/huggingface/transformers
    and https://github.com/rxn4chemistry/rxnfp.

    Args:
        vocab_file: path to a token per line vocabulary file.
            Example:
                input: "[H]C([H])([H])C([H])([.>-][Br->0])(C([H])([H])N([H])[H])[->.][Cl0>-]"
                single: "[H] C ( [H] ) ( [H] ) C ( [->=] C ( C # N ) ( C # N ) [->.] [Br0>-] ) ( [->.] [H] [.>-] [Cl->0] ) C # N"
                multiple: "[H] C ( [H] ) ( [H] ) C ( [ - > = ] C ( C # N ) ( C # N ) [ - > . ] [ Br 0 > - ] ) ( [ - > . ] [H] [ . > - ] [ Cl - > 0 ] ) C # N"
                modified:   "[H] C ( [H] ) ( [H] ) C ( { - > = } C ( C # N ) ( C # N ) { - > . } { Br 0 > - } ) ( { - > . } [H] { . > - } { Cl - > 0 } ) C # N"
    """  # noqa E501

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        do_lower_case=False,
        **kwargs,
    ) -> None:
        """Constructs an SmilesTokenizer.
        Args:
            vocabulary_file: vocabulary file containing tokens.
            unk_token: unknown token. Defaults to "[UNK]".
            sep_token: separator token. Defaults to "[SEP]".
            pad_token: pad token. Defaults to "[PAD]".
            cls_token: cls token. Defaults to "[CLS]".
            mask_token: mask token. Defaults to "[MASK]".
        """
        self.create_file(vocab_file)
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            do_lower_case=do_lower_case,
            **kwargs,
        )

    def base_tokenizer(self, smiles: str) -> list[str]:
        """
        Tokenizes a SMILES molecule or reaction.
        """
        smiles = self.custom_transform_hook(smiles)
        return [token for token in self.pattern.findall(smiles)]

    @property
    def vocab_list(self) -> list[str]:
        """List vocabulary tokens.
        Returns:
            a list of vocabulary tokens.
        """
        return list(self.vocab.keys())

    def _tokenize(self, text: str) -> str:
        """call the tokenizer from the Tokenizer class.
        Args:
            text: text to tokenize.
        Returns:
            extracted tokens.
        """

        return self.base_tokenizer(text)

    def custom_transform_hook(self, smiles: str) -> str:
        """
        A hook to modify the smiles before tokenization, to be reimplemented by subclasses if needed.

        Args:
            smiles: a SMILES string.

        Returns:
            a modified SMILES string.
        """
        return smiles

    def create_file(self, file_path: str) -> str:
        """
        If the file does not exist, create an empty file.

        Args:
            file_path (str): path to the file.
        """
        if not os.path.exists(file_path):
            with open(file_path, "w") as file:
                file.write("")
