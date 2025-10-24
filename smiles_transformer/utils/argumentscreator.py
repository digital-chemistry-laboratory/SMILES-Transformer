import ast
import json
import os
import sys

import yaml

from smiles_transformer.utils.path_finder import path_finder


class ArgumentsCreator:
    def __init__(self):
        """
        This script is used to create the arguments.txt file.
        It will ask you for the values of each argument and save them in the file.
        If the file already exists, it will ask you if you want to purge it or append to it.
        """
        with open(
            path_finder("./configurations/config.yaml", is_file=True), "r"
        ) as stream:
            self.params = yaml.safe_load(stream)
        self.new_params = {}
        purge = input("Would you like to purge the current file? [y/N]")
        print(f"Answer: {purge}")
        self.arguments_path = path_finder(
            "./configurations/arguments.txt", is_file=True
        )
        if purge.lower() == "y":
            with open(self.arguments_path, "w") as f:
                f.write("")

    def save_params(self, final_exit=True):
        """
        Saves the parameters in the file.
        """
        with open(self.arguments_path, "a") as f:
            f.write("\n")
            json.dump(self.new_params, f)
        if final_exit:
            sys.exit()
        self.new_params = {}

    def select_category(self, parameters, subcategory=False):
        """
        Asks the user to select a category from the parameters.
        """
        print("Categories:")
        print("0)  Exit and save")
        print("100) Save this line and begin a new one")
        if subcategory:
            print("101) Go back general category selection\n")
        else:
            print("")
        for i, category in enumerate(parameters):
            print(f"{i+1})  {category}")
        selected_index = int(input("\nSelect number and press enter:"))
        if selected_index == 0:
            self.save_params(final_exit=True)
        if selected_index == 100:
            self.save_params(final_exit=False)
            return None
        if selected_index == 101:
            return None
        return list(parameters.keys())[selected_index - 1]

    def print_arguments(self):
        """
        prints the current arguments
        """
        try:
            os.system("clear")  # this is just for linux...
        except:  # noqa E722
            pass
        print("\nCurrent already written arguments:")
        with open(self.arguments_path, "r") as f:
            print(f.read())
        print("\n")
        print("New arguments:")
        print(self.new_params)
        print("\n")

    def run(self):
        """
        Runs the script.
        """
        while True:
            self.print_arguments()
            category = self.select_category(self.params)
            if category is None:
                continue
            subcategory = self.select_category(self.params[category], subcategory=True)
            if subcategory is None:
                continue
            value = input(f"Enter value for {subcategory}:")
            try:
                # Try to evaluate the input string as a Python literal expression
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If evaluation fails, fall back to treating the input as a string
                pass

            if category not in self.new_params:
                self.new_params[category] = {}
            self.new_params[category][subcategory] = value


if __name__ == "__main__":
    ArgumentsCreator().run()
