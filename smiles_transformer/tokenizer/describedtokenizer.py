from .basetokenizertemplate import BaseTokenizerTemplate
import re


class DescribedTokenizer(BaseTokenizerTemplate):
    pattern = re.compile(
        r"(<.*?>|%%%%|\[.*?:|BIN_[0-9]*\]|[.]|[A-Z][a-z]|\[|\]|[A-Z]|[a-z]|\(|\)|=|#|-|\+|\\+|\/|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]{1,2}|\^)"  # noqa 605
    )
    tokenizer_kind = "described_tokenizer"


if __name__ == "__main__":
    tokenizer = DescribedTokenizer(vocab_file="data/vocab/vocab_described.txt")
    print(isinstance(tokenizer, BaseTokenizerTemplate))
