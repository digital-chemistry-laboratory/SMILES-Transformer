import os


def path_finder(
    prefix: str,
    path_from_source: str,
    is_file: bool = False,
    create_path_if_unavailable: bool = False,
) -> str:
    """
    Join prefix + path_from_source into a clean absolute path.

    - A leading slash or backslash in path_from_source is stripped off,
      so "/config.yaml" → "config.yaml" (i.e. relative).
    - If path_from_source *still* looks like an absolute path (drive-letter
      or UNC), we use it as-is.
    - Otherwise we join it to prefix (made absolute first), normalize,
      and optionally mkdir if it's a directory.
    """

    # 1) strip any leading slash/backslash so "/config.yaml" won't be
    #    interpreted as root of C:
    path_rel = path_from_source.lstrip("/\\")

    # 2) if it's now a true absolute path (e.g. "C:\foo\bar" or "\\\\share\x"),
    #    use it directly.  Otherwise join to prefix:
    if os.path.isabs(path_rel):
        absolute_path = os.path.normpath(path_rel)
    else:
        base = os.path.abspath(prefix)
        absolute_path = os.path.normpath(os.path.join(base, path_rel))

    # 3) if this is a directory and we should create it, do so,
    #    and ensure it ends with a trailing separator:
    if not is_file:
        if create_path_if_unavailable and not os.path.exists(absolute_path):
            os.makedirs(absolute_path, exist_ok=True)
        absolute_path = absolute_path.rstrip(os.sep) + os.sep

    return absolute_path
