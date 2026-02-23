from .basetokenizertemplate import BaseTokenizerTemplate
import re


class IndividualTokenizer(BaseTokenizerTemplate):
    pattern = re.compile(
        r"([.]|[A-Z][a-z]|\[|\]|[A-Z]|[a-z]|\(|\)|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]|\^|{|}|\|)"  # noqa 605
    )
    tokenizer_kind = "individual"

    def custom_transform_hook(self, smiles: str) -> str:
        """
        Custom transformation hook for the IndividualTokenizer.
        This method is used to replace the period in bond changes to a dollar sign.

        Args:
            smiles (str): A SMILES string.

        Returns:
            str: The transformed SMILES string.
        """
        custom_pattern = re.compile(r"(\[[\.\-=#\:]>[\.\-=#\:]\])")
        smiles = re.sub(
            custom_pattern,
            lambda match: match.group(0).replace(".", "$"),
            smiles,
        )
        return smiles


if __name__ == "__main__":
    tokenizer = IndividualTokenizer(
        vocab_file="/home/giustino/git/smiles-cgr-transformers/data/interim/vocab_individual.txt"
    )
    print(isinstance(tokenizer, BaseTokenizerTemplate))
