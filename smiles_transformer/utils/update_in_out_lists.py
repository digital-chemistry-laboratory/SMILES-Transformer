def update_in_out_lists(
    in_list: list, out_list: list, out_name: str
) -> tuple[list, list]:
    """
    Updates input and output lists for processing data.

    Appends the last element of `out_list` to `in_list` and appends `out_name`
    to `out_list`.

    Args:
        in_list (list): The list of input features.
        out_list (list): The list of output features.
        out_name (str):  The name of the new output feature to add.

    Returns:
        tuple[list, list]: The updated input list and output list.
    """

    in_list.append(out_list[-1])
    out_list.append(out_name)
    return in_list, out_list
