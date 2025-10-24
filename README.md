# Getting Started

## Installation

First, create and activate a new conda environment:

```bash
conda create -n smiles_transformer python=3.13
conda activate smiles_transformer
```

To install the package, just run:

```bash
git clone https://github.com/your-username/smiles_transformer.git
cd smiles_transformer
pip install .
```

If the install comand fails because of CGRTools, you need to install a C toolchain:

```bash
sudo apt update && sudo apt install -y build-essential
```

And try the `pip install` command again.

## Quick Start

Here is a quick example of how to use the library:

- Create a [WandB](https://wandb.ai/) account and login locally.
- Modify the config file found at `configurations/config.yaml` to suit your use-case. The complete documentation for the config
  file can be found here: [configurations](configurations/configurations.md)
    - The provided configuration file should already be working for a quick run after adjusting:
        - `path_to_outputs`: path to the output folder where models should be saved
        - `path_to_vocab_folder`: path where the vocabulary files are/should be located located. (ex. <path_to_root>/data/vocab)
        - `dataset_folder`: the path to the *folder* in which the dataset is in, see below for an example dataset
        - `dataset_name`: the name of the file
        - `column_name`: the name of the column containing the reactions
        - `label_column_name`: only for regression or classification, the name of the column containing the labels.
        - `mapping`: if using SMILES and they are atom-mapped, you should set this to "remove", otherwise for SMILES/CGR set this to null.
        - `convert_to_cgr`: If using SMILES/CGR, set this to True. If using SMILES, set this to False.
    - Note: `eval_size` should not be too big (~<1000) as it will cause great performance issues. Should you set it as a
      fraction of the dataset, do ensure that the resulting size is not too big.
    - Generally, you should first pre-train a model by setting `model_mode` to `pretraining`, then finetune it by either
      setting the `model_mode` to `regression` or `classification`. When fine-tuning, set the `run_name` to whatever the
      WandB-set run name of the pre-training run was.
    - the data should be in a csv file as in this example: [e2sn2.csv](https://github.com/hesther/reactiondatabase/blob/main/data/e2sn2.csv)
- Once done, run the main file to train your model:

```python
python -m smiles_transformer.main "configurations/"
```

Alternatively, run a cross-validation run:

```python 
python -m smiles_transformer.launch_cv "configurations/"
```

To run a prediction, leave the config file as is except for the `dataset_settings` where the `dataset_folder` and
`dataset_name` should point to the csv where your reactions/molecules are in and `column_name` to point to the column
where your molecules/reactions to predict are in. `run_name` should be the name of a finetuned run. Then just run:

```python 
python -m smiles_transformer.predict "configurations/" "path/to/desired/outputs/file/with/predictions.csv"
```
The output will be in the prediction column.