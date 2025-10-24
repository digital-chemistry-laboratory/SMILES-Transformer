from smiles_transformer.preprocessing.transform.transformtemplate import (
    TransformTemplate,
)


# TODO: add TESTS
class ContentFilter(TransformTemplate):
    """
    Filter out datapoints with a pattern (ex: if [H] is present) in a certain column.
    This class needs a kwargs argument pattern_to_filter_out for the pattern and pattern_filter_column for the column to look at.
    """

    no_pattern_to_filter_out_error = "pattern_to_filter_out not in given. Please provide pattern_to_filter_out in kwargs of the class init or the step function."
    no_with_without_error = (
        "pattern_to_filter_out should begin with 'with:' or 'without:'"
    )

    def step(self, batch):
        assert (
            "pattern_to_filter_out" in self.kwargs
        ), self.no_pattern_to_filter_out_error
        assert "with" in self.kwargs.get(
            "pattern_to_filter_out"
        ), self.no_with_without_error

        if "with:" in self.kwargs.get("pattern_to_filter_out").lower().strip().replace(
            " ", ""
        ):
            pattern = self.kwargs.get("pattern_to_filter_out").replace("with", "")
            pattern = pattern.replace(":", "").strip()
            return batch[batch[self.in_column].str.contains(pattern, case=True)]
        if "without:" in self.kwargs.get(
            "pattern_to_filter_out"
        ).lower().strip().replace(" ", ""):
            pattern = self.kwargs.get("pattern_to_filter_out").replace("without", "")
            pattern = pattern.replace(":", "").strip()
            return batch[~batch[self.in_column].str.contains(pattern, case=True)]

    @property
    def init_message(self) -> str:
        return f"Selecting only datapoints: \x1b[6;30;42m{self.kwargs['pattern_to_filter_out']}\x1b[0m"
