from .basetokenizertemplate import BaseTokenizerTemplate
import re


class SingleTokenizer(BaseTokenizerTemplate):
    pattern = re.compile(
        r"([A-Za-z]{1,2}H\^{1,2}|{.\|.}|\%\([0-9]{3}\)|[\*\^]>[\*\^]|[\.\-\=#:]>[\.\-\=#:]|[\+\-]*[0-9]?>[\+\-]*[0-9]?|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]|\[[A-Z]*[a-z]*[\*\^]*[1-9]*[\+\-]*\]|\[|\]|}|{)"  # noqa 605
    )
    tokenizer_kind = "single"

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
    tokenizer = SingleTokenizer(
        vocab_file="/home/giustino/git/smiles-cgr-transformers/data/interim/vocab_single.txt"
    )
    print(tokenizer.tokenizer_kind)
