from setuptools import setup, find_packages

setup(
    name="smiles_transformer",
    author="Giustino Sulpizio",
    version="0.1",
    packages=find_packages(),  # Finds the package in the root
    include_package_data=True,
    package_data={
        "smiles_transformers": ["configurations/*.json"],
    },
    install_requires=[
        "PyYAML>=5.1",
        "cgrtools>=4.1.35",
        "pandas>=2.2.3",
        "rxnmapper @ git+https://github.com/gSulpizio/rxnmapper.git@main#egg=rxnmapper",
        "sr-smiles @ git+https://github.com/heid-lab/sr-smiles",
        "torch>=2.6.0",
        "transformers>=4.49.0",
        "datasets>=3.3.2",
        "matplotlib>=3.10.1",
        "rdkit>=2024.9.5",
        "wandb>=0.19.7",
        "tqdm>=4.67.1",
        "scikit-learn>=1.6.1",
        "tabulate>=0.9.0",
        "accelerate>=1.4.0",
        "pyperclip>=1.9.0",
        "tqdm>=4.67.1",
        "swifter==1.4.0",
        "py-mini-racer==0.6.0",
    ],
)
