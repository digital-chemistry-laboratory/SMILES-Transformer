import json
import sys

from smiles_transformer.main import main


def arg_parser(parameters):
    """
    Wrapper for main function, parses arguments given by the shell.
    To use this, call it as follows:

    python -m smiles_transformer.arg_parser '{"settings": {"subsettings":"value"}}'
    """
    parameters_str = sys.argv[1]
    parameters = json.loads(parameters_str)
    main(parameters)


if __name__ == "__main__":
    arg_parser(None)
