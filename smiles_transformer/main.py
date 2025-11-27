import yaml
import os
import wandb
import sys
from smiles_transformer.dataset import *
from smiles_transformer.model import *
from tabulate import tabulate
from smiles_transformer.preprocessing import OnTheFlyDataLoader, PremadeDataLoader
from smiles_transformer.preprocessing.transform import *
from smiles_transformer.utils.vocab import Vocab
from smiles_transformer.trainer import *
from smiles_transformer.utils import path_finder, update_in_out_lists
from smiles_transformer.splitter import (
    SKLearnSplitter,
    AugmentCVLabeledSplitter,
    CVLabeledSplitter,
)
from smiles_transformer.evaluation import *
from smiles_transformer.tokenizer import *
from datetime import datetime
import json
import random
import numpy as np

def preprocess(params):
    """
    Preprocesses the dataset and loads the vocabulary.

    Args:
        params (dict): Dictionary containing the parameters for the model.

    Returns:
        tuple: Tuple containing the dataset, the vocabulary, the splitter, the transform list, the out column list.
    """
    n_augmentations = None

    transforms = []
    transform_list = []
    train_test_splitter = SKLearnSplitter  # default way of train test splitting.
    in_column_list = []
    out_column_list = [params["dataset_settings"]["column_name"]]

    assert (
        (params["dataset_settings"]["n_datapoints"] is None)
        or (params["dataset_settings"]["augment"] is None)
        or (params["general_settings"]["test_mode"] is True)
    ), "You can't use n_points and augment at the same time outside of test mode because this leads to some difficulties."

    if params["general_settings"]["verbose"]:
        print("Loading preprocessing transforms and filters")

    transforms.append(EmptyTransform)
    transform_list.append("empty_transform")
    in_column_list, out_column_list = update_in_out_lists(
        in_column_list, out_column_list, out_column_list[-1]
    )

    if params["dataset_settings"]["fix_nitro"]:
        # clean reactions that have erroneous nitro groups
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, out_column_list[-1]
        )
        transform_list.append("fix_nitro")

        transforms.append(CorrectNitroTransform)

    if "remove" in (params["dataset_settings"]["mapping"] or []):
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "unmapped"
        )
        transform_list.append("remove_atom_mapping")

        transforms.append(RemoveAtomMappingTransform)
    if "smiles" in (params["dataset_settings"]["augment"] or []):
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, out_column_list[-1]
        )

        n_augmentations = int(
            params["dataset_settings"]["augment"].replace("smiles:", "")
        )
        transform_list.append(f"augment_smiles:{n_augmentations}")

        transforms.append(AugmentSMILESTransform)
    if "remap" in (params["dataset_settings"]["mapping"] or []):
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "unmapped"
        )
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "remapped"
        )

        transform_list.append("rxn_mapper")
        transforms.append(RemoveAtomMappingTransform)
        transforms.append(RXNMapTransform)
    if params["dataset_settings"]["remap_confidence_threshold"]:
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list,
            out_column_list,
            params["dataset_settings"]["remap_confidence_column_name"],
        )
        in_column_list.append(in_column_list[-1])
        out_column_list.append(out_column_list[-2])

        transform_list.append("confidence_threshold")
        transform_list.append("empty_transform")

        transforms.append(MappingConfidenceFilter)
        transforms.append(EmptyTransform)
    if params["dataset_settings"]["invert_reactions"]:
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "inverted_reaction"
        )
        in_column_list.append(in_column_list[-1])
        out_column_list.append(out_column_list[-1])
        transform_list.append("reaction_inversion")
        transforms.append(ReactionInversionTransform)
        transform_list.append("label_sign_inversion")
        transforms.append(LabelSignInversionTransform)
    if (
        params["dataset_settings"]["explicify_hydrogens"]
        and params["dataset_settings"]["convert_to"] != "SMILES/CGR"
    ):
        # Try and explicify hydrogens using RDKit if possible, otherwise return identical batch
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, out_column_list[-1]
        )
        transform_list.append("Explicify Hydrogens")
        transforms.append(ExplicifyHydrogensTransform)

    if params["dataset_settings"]["convert_to"] == "sr-SMILES":
        # Try and explicify hydrogens using RDKit if possible, otherwise return identical batch
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "sr-SMILES"
        )
        transform_list.append("convert_to_sr-SMILES")
        transforms.append(SMILEStoSRSMILESTransform)

    if params["dataset_settings"]["convert_to"] == "SMILES/CGR" or (
        "cgr" in (params["dataset_settings"]["augment"] or [])
    ):
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "CGR"
        )
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "CGR"
        )
        if "cgr" in (params["dataset_settings"]["augment"] or []):
            in_column_list, out_column_list = update_in_out_lists(
                in_column_list, out_column_list, out_column_list[-1]
            )
            transform_list.append("smiles_to_cgr_conversion")
            n_augmentations = int(
                params["dataset_settings"]["augment"].replace("cgr:", "")
            )
            transform_list.append(f"augment_cgr:{n_augmentations}")
        transforms.append(SMILEStoCGRTransform)
        transforms.append(CGRtoStrTransform)
    if params["dataset_settings"]["n_max_bond_changes"]:
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "n_CGR_changes"
        )
        in_column_list.append(in_column_list[-1])
        out_column_list.append(out_column_list[-2])

        transform_list.append("max_bond_changes")
        transform_list.append("empty_transform")
        transforms.append(MaxCGRChangeFilter)
        transforms.append(EmptyTransform)
    if params["dataset_settings"]["n_min_bond_changes"]:
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "n_CGR_changes"
        )
        in_column_list.append(in_column_list[-1])
        out_column_list.append(out_column_list[-2])
        transform_list.append("min_bond_changes")
        transform_list.append("empty_transform")
        transforms.append(MinCGRChangeFilter)
        transforms.append(EmptyTransform)

    if params["dataset_settings"]["pattern_to_filter_out"]:
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, out_column_list[-1]
        )
        transform_list.append("content_filter")
        transforms.append(ContentFilter)
    if params["training_settings"]["model_mode"] == "pretraining":
        params["general_settings"]["tags"].append("pretraining")

    if (
        params["descriptors_settings"]["atomic_descriptors"]
        or params["descriptors_settings"]["molecular_descriptors"]
    ):
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, out_column_list[-1]
        )
        transform_list.append("parse_descriptor_lists")
        transforms.append(ListMatrixStringFixer)

    if params["descriptors_settings"]["descriptors_to_bin"]:
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, out_column_list[-1]
        )
        transform_list.append("descriptor_binner")
        transforms.append(DescriptorBinner)

    if params["descriptors_settings"]["atomic_descriptors"]:
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "descriptor_augmented_smiles"
        )
        transform_list.append("atom_inserted")
        transforms.append(AtomDescriptorInserter)

    if params["descriptors_settings"]["molecular_descriptors"]:
        in_column_list, out_column_list = update_in_out_lists(
            in_column_list, out_column_list, "descriptor_augmented_smiles"
        )
        transform_list.append("mol_inserted")
        transforms.append(MolecularDescriptorInserter)

    out_column_list.pop(0)

    if params["general_settings"]["verbose"]:
        print(
            f"Loaded transforms and filters.\n preprocessing consists of:\n{transform_list}\n"
        )

    if n_augmentations and type(params["test_settings"]["test_size"]) in [
        int,
        float,
    ]:
        train_test_splitter = AugmentCVLabeledSplitter
    if type(params["test_settings"]["test_size"]) is str:
        train_test_splitter = CVLabeledSplitter

    labeler, splitter = train_test_splitter(
        params["general_settings"]["random_state"]
    ).create()

    tokenizer_dict = {
        "single": SingleTokenizer,
        "individual": IndividualTokenizer,
        "described": DescribedTokenizer,
        "BPE": BPETokenizer,
    }
    tokenizer = tokenizer_dict[params["general_settings"]["tokenizer_kind"]]

    dataloader_class = OnTheFlyDataLoader
    if params["test_settings"]["n_folds"] == "premade":
        dataloader_class = PremadeDataLoader

    dataloader = dataloader_class(
        path=params["dataset_settings"]["dataset_folder"],
        dataset_name=params["dataset_settings"]["dataset_name"],
        transforms=transforms,
        labeler=labeler,
        test_size=params["test_settings"]["test_size"],
        eval_size=params["test_settings"]["eval_size"],
        n_points=params["dataset_settings"]["n_datapoints"],
        verbose=params["general_settings"]["verbose"],
        shuffle=params["general_settings"]["random_state"] is None,
        remap_confidence_threshold=params["dataset_settings"][
            "remap_confidence_threshold"
        ],
        n_max_bond_changes=params["dataset_settings"]["n_max_bond_changes"],
        n_min_bond_changes=params["dataset_settings"]["n_min_bond_changes"],
        explicify_hydrogens=params["dataset_settings"]["explicify_hydrogens"],
        n_augmentations=n_augmentations,
        label_column_name=params["dataset_settings"]["label_column_name"],
        pattern_to_filter_out=params["dataset_settings"]["pattern_to_filter_out"] or [],
        descriptors_to_bin=params["descriptors_settings"]["descriptors_to_bin"],
        num_bins=params["descriptors_settings"]["num_bins"],
        binning_strategy=params["descriptors_settings"]["binning_strategy"],
        atomic_descriptors=params["descriptors_settings"]["atomic_descriptors"],
        molecular_descriptors=params["descriptors_settings"]["molecular_descriptors"],
        original_column_name=params["dataset_settings"]["column_name"],
        mapping=params["dataset_settings"]["mapping"],
        sr_smiles_settings=params["dataset_settings"]["sr_smiles_settings"],
    )

    if params["general_settings"]["verbose"]:
        print("DataLoader initialized. Loading dataset.")

    X = dataloader.load_dataset(
        in_column_list=in_column_list,
        out_column_list=out_column_list,
        separator=params["dataset_settings"]["separator"],
    )
    if params["general_settings"]["verbose"]:
        print(f"Dataset Loaded: {X.shape[0]} SMILES\n Loading Vocab.")
        print(tabulate(X.head(), headers="keys"))

    vocab = Vocab(
        tokenizer,
        path_to_vocab_folder=params["vocabulary_settings"]["path_to_vocab_folder"],
        sampling=params["vocabulary_settings"]["sampling"],
        sample_size=params["vocabulary_settings"]["sample_size"],
        max_vocab_size=params["vocabulary_settings"]["max_vocab_size"],
        min_frequency=params["vocabulary_settings"]["min_frequency"],
    )
    if params["vocabulary_settings"]["generate_vocab"] and (
        params["training_settings"]["model_mode"] == "pretraining"
    ):
        vocab.autogenerate(
            X[out_column_list[-1]].to_numpy(),
            params["vocabulary_settings"]["generate_vocab"],
        )
    if params["general_settings"]["verbose"]:
        print(f"Vocab Loaded: {vocab.size} tokens")

    if params["vocabulary_settings"]["terminate_after_vocab_creation"]:
        print("Vocab created. Terminating.")
        sys.exit()
    if params["general_settings"]["verbose"]:
        print("Vocab Loaded. Preparing Trainer.")

    return (
        X,
        vocab,
        tokenizer,
        splitter,
        transform_list,
        out_column_list,
    )




def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CRITICAL: These two lines make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 1.8
    torch.use_deterministic_algorithms(True, warn_only=True)


def train(
    params,
    X,
    vocab,
    tokenizer,
    splitter,
    transform_list,
    out_column_list,
):
    """
    Trains the model.

    Args:
        params (dict): Dictionary containing the parameters for the model.
        X (pandas.DataFrame): DataFrame containing the dataset.
        vocab (Vocab): Vocabulary object.
        splitter (Splitter): Splitter object.
        transform_list (list): List of transforms.
        out_column_list (list): List of output columns.
        original_smiles_column (str): Original SMILES column.

    Returns:
        tuple: Tuple containing the model and the trainer.
    """

    init_dict = {
        "pretraining": (
            PretrainingTrainerFactory,
            PretrainingModelFactory,
            PretrainingDatasetFactory,
        ),
        "regression": (
            FinetuningTrainerFactory,
            FinetuningModelFactory,
            RegressionDatasetFactory,
        ),
        "classification": (
            FinetuningTrainerFactory,
            FinetuningModelFactory,
            ClassificationDatasetFactory,
        ),
    }
    trainer_factory, model_factory, dataset_factory = init_dict[
        params["training_settings"]["model_mode"]
    ]
    if params["dataset_settings"]["load_dataset_path"]:
        dataset_factory = PreprocessedDatasetFactory

    if len(out_column_list) > 0:
        features_column_name = out_column_list[-1]
    else:
        features_column_name = params["dataset_settings"]["column_name"]
    n_labels = 1
    if params["training_settings"]["model_mode"] == "classification":
        n_labels = len(X[params["dataset_settings"]["label_column_name"]].unique())
    trainer = trainer_factory(
        dataset_factory=dataset_factory,
        path_to_outputs=params["general_settings"]["path_to_outputs"],
        dataset_folder=params["dataset_settings"]["dataset_folder"],
        model_factory=model_factory,
        dataset_name=params["dataset_settings"]["dataset_name"],
        features_column_name=features_column_name,
        path_to_vocab_file=vocab.path_to_vocab_file,
        smilestokenizer=tokenizer,
        splitter=splitter,
        label_column_name=params["dataset_settings"]["label_column_name"],
        model_type=params["training_settings"]["model_type"],
        run_name=params["training_settings"]["run_name"],
        gradient_accumulation_steps=params["training_settings"][
            "gradient_accumulation_steps"
        ],
        mask_prob=params["training_settings"]["mask_p"],
        learning_rate=params["training_settings"]["learning_rate"],
        test_size=params["test_settings"]["test_size"],
        eval_size=params["test_settings"]["eval_size"],
        hidden_dropout=params["training_settings"]["hidden_dropout"],
        eval_steps=params["training_settings"]["eval_steps"],
        model_size=params["training_settings"]["model_size"],
        early_stopping_patience=params["training_settings"]["early_stopping_patience"],
        early_stopping_threshold=params["training_settings"][
            "early_stopping_threshold"
        ],
        test_mode=params["general_settings"]["test_mode"],
        verbose=params["general_settings"]["verbose"],
        num_train_epochs=params["training_settings"]["num_train_epochs"],
        num_train_steps=params["training_settings"]["num_train_steps"],
        scale_target=params["dataset_settings"]["scale_target"],
        max_length=params["training_settings"]["max_length"],
        batch_size=params["training_settings"]["batch_size"],
        auto_find_batch_size=params["training_settings"]["auto_find_batch_size"],
        fp16=params["training_settings"]["fp16"],
        warmup_ratio=params["training_settings"]["warmup_ratio"],
        additional_features=params["dataset_settings"]["additional_features"] or [],
        n_layers=params["training_settings"]["n_layers"],
        n_labels=n_labels,
        layer_width=params["training_settings"]["layer_width"],
        logging_steps=params["training_settings"]["logging_steps"],
        checkpoint_number=params["training_settings"]["checkpoint_number"],
        save_total_limit=params["training_settings"]["save_total_limit"],
        n_trials=None,  # params["training_settings"]["n_trials"],
        random_state=params["general_settings"]["random_state"],
        save_dataset_path=params["dataset_settings"]["save_dataset_path"],
        load_dataset_path=params["dataset_settings"]["load_dataset_path"],
        params=params,
    )
    if params["general_settings"]["verbose"]:
        print("Trainer initialized. Training model.")
    model = trainer.create(dataset=X)
    if params["general_settings"]["verbose"]:
        print("Model trained. Evaluating model if in finetuning mode.")
    with open(
        os.path.join(
            params["general_settings"]["path_to_outputs"],
            params["training_settings"]["model_mode"],
            params["training_settings"]["model_type"],
            params["training_settings"]["model_size"],
            params["general_settings"]["tokenizer_kind"],
            "overall_config.json",
        ),
        "w",
    ) as f:
        f.write(json.dumps(params, indent=4))
    return model, trainer


def evaluate(params, model, trainer):
    """
    Evaluates the model.

    Args:
        params (dict): Dictionary containing the parameters for the model.
        model (Model): Model object.
        trainer (Trainer): Trainer object.

    Returns:
        dict: Dictionary containing the evaluation results.
    """
    mean = 0
    std = 1
    if params["dataset_settings"]["scale_target"] and params["training_settings"][
        "model_mode"
    ] not in ["pretraining"]:
        mean = trainer.dataset_factory.mean
        std = trainer.dataset_factory.std
    model_evaluation = None
    if params["training_settings"]["model_mode"] == "regression":

        evaluator = RegressionEvaluator(
            model,
            output_dir=trainer.output_dir,
            wandb_run=trainer.run,
            verbose=params["general_settings"]["verbose"],
            std=std,
            mean=mean,
            target_median=trainer.dataset_factory.target_median,
        )

    if params["training_settings"]["model_mode"] == "classification":
        evaluator = ClassificationEvaluator(
            model,
            output_dir=trainer.output_dir,
            wandb_run=trainer.run,
            verbose=params["general_settings"]["verbose"],
            scale_target=params["dataset_settings"]["scale_target"],
            label_encoder=trainer.dataset_factory.label_encoder,
            label_counter=trainer.dataset_factory.label_counter,
        )
    if params["training_settings"]["model_mode"] != "pretraining":

        model_evaluation = evaluator.evaluate(
            trainer.test_set, trainer.tokenized_dataset
        )

    if params["general_settings"]["verbose"]:
        print("Logging to wandb. Finishing run.")
    trainer.run.tags = trainer.run.tags + (
        params["training_settings"]["model_mode"],
        params["training_settings"]["model_type"],
        params["training_settings"]["model_size"],
        params["general_settings"]["tokenizer_kind"],
    )
    if "sweep" in trainer.run.name:
        trainer.run.tags = trainer.run.tags + ("sweep",)

    # Test the model on the test set:
    trainer.run.config.update(params, allow_val_change=True)
    # an if statement to check if in the params the model should be saved on wandb
    if (not params["general_settings"]["test_mode"]) and (
        params["general_settings"]["save_model_to_wandb"]
    ):
        path_to_model = path_finder(
            prefix=params["general_settings"]["path_to_outputs"],
            path_from_source=f"outputs/{params['training_settings']['model_mode']}/{params['training_settings']['model_type']}/{params['training_settings']['model_size']}/{params['general_settings']['tokenizer_kind']}/{trainer.run.name}",
            is_file=False,
            create_path_if_unavailable=True,
        )

        art = wandb.Artifact(
            f'{params["training_settings"]["model_mode"]}-{params["general_settings"]["tokenizer_kind"]}-{wandb.run.id}',
            type="model",
        )
        art.add_dir(path_to_model)
        wandb.log_artifact(art, aliases=["latest"])
    return model_evaluation


def check_parameters(params):
    assert params["training_settings"]["model_mode"] in [
        "regression",
        "classification",
        "pretraining",
    ], "model_mode must be either pretraining, regression, or classification"
    if params["training_settings"]["model_mode"] == "classification":
        assert (
            params["dataset_settings"]["scale_target"] is False
        ), "If the model mode is classification, target scaling must be False."
    if params["training_settings"]["model_mode"] not in ["pretraining"]:
        assert (
            params["training_settings"]["run_name"] is not None
        ), "if the model is not pretraining, a run name must be provided in the configuration file."
    # vocab_path = path_finder(prefix=params["vocabulary_settings"]["path_to_vocab_folder"],path_from_source=f"vocab_{params['general_settings']['tokenizer_kind']}.txt",is_file=True,)

    assert os.path.exists(
        params["vocabulary_settings"]["path_to_vocab_folder"]
    ), f"Vocabulary file not found at {params['vocabulary_settings']['path_to_vocab_folder']}.Verify that the path is correct, and that the file exists!"

    return


def main(path_to_config_folder, alternative_config=None):
    """
    Main function of the project. Loads the dataset, the vocabulary, and trains the model.

    Args:
        alternative_config (dict): Dictionary containing the parameters to overwrite in the config.yaml file. Default: {}.

    Returns:
        None
    """

    # prepare which factory is used for each factory type

    print(f"Starting main function at {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    with open(
        path_finder(
            prefix=path_to_config_folder, path_from_source="/config.yaml", is_file=True
        ),
        "r",
    ) as stream:
        params = yaml.safe_load(stream)
    check_parameters(params)
    if params["general_settings"]["verbose"]:
        print("Config Loaded")

    if alternative_config is None:
        alternative_config = {}
    for category in alternative_config:
        params[category].update(alternative_config[category])
    if params["general_settings"]["random_state"] is not None:
        seed_everything(params["general_settings"]["random_state"])
    (
        X,
        vocab,
        tokenizer,
        splitter,
        transform_list,
        out_column_list,
    ) = preprocess(params)

    model, trainer = train(
        params,
        X,
        vocab,
        tokenizer,
        splitter,
        transform_list,
        out_column_list,
    )

    model_evaluation = evaluate(params, model, trainer)
    trainer.run.config.update(
        {
            "dataset_size": len(X),
            "num_parameters": model.model.num_parameters(),
            "transform_list": transform_list,
        },
        allow_val_change=True,
    )

    if (type(params["test_settings"]["test_size"]) is str) and (
        "/" in params["test_settings"]["test_size"]
    ):
        return model_evaluation

    trainer.run.finish()
    if params["general_settings"]["verbose"]:
        print("EOF")
    return model_evaluation


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the smiles_transformer model training and evaluation."
    )
    parser.add_argument(
        "config_folder",
        type=str,
        help="Path to the configuration folder (where config.yaml is located).",
    )
    # If you want to support additional optional arguments, add them here:
    # parser.add_argument("--other", type=str, help="Some other argument")

    args = parser.parse_args()
    main(args.config_folder)
