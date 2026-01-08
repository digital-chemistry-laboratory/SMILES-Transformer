#########################
Configuration File
#########################

The entire training, testing, and data processing workflow is controlled by a single ``config.yaml`` file. This page provides a comprehensive reference for every available parameter, organized by section.

---

General Settings
================

These settings control the overall behavior of the script, such as logging, execution modes, and paths.

**verbose**
  Controls the amount of information printed to the console during execution.
  - **Type**: ``boolean``
  - **Options**:
    - ``True``: Print detailed status updates.
    - ``False``: Run silently with minimal output.

**test_mode**
  Enables or disables WandB logging.
  - **Type**: ``boolean``
  - **Options**:
    - ``True``: Disable WandB logging.
    - ``False``: Enable WandB logging.

**tokenizer_kind**
  Specifies the tokenization strategy to be used. ``"single"`` and ``"individual"`` are identical for SMILES, but change for for SMILES/CGR:
  A bond token [.>-] will be tokenized as ``"[.>-]"`` with ``"single"`` and ``"[,.,>,-,]"`` with ``"individual"``.
  - **Type**: ``string``
  - **Options**:
    - ``"single"``: Treats every atomic symbol as a tokens.
    - ``"individual"``: Tokenizes the input into individual characters or components.
    - ``"described"``: A custom strategy integrates atomic descriptors as bins in the string for SMILES.


**random_state**
  Seed for random number generators to ensure reproducibility in data splitting and other stochastic processes. Disable by setting to ``null``
  - **Type**: ``integer`` or ``null``
.. warning::
  Setting a fixed integer value helps in achieving reproducible results. However, this does not guarantee bit-for-bit identical model initialization across all hardware and library versions. Set to ``null`` for general use.

**save_model_to_wandb**
  If ``True``, the final trained model artifacts will be uploaded to Weights & Biases at the end of the run.
  - **Type**: ``boolean``

**tags**
  A list of tags to associate with the run in Weights & Biases for easier filtering and organization.
  - **Type**: ``list``
  - **Default**: ``[]``
  - **Example**: ``[ "BERT", "pretraining", "v2" ]``

**path_to_outputs**
  The absolute path to the root folder where all model outputs (checkpoints, logs, etc.) will be saved.
  - **Type**: ``string``

---

Test Settings
=============

Parameters for configuring the validation and test dataset splits, as well as cross-validation.

**test_size**
  The proportion or absolute number of the dataset to reserve for the test set.
  - **Type**: ``float``, ``integer``
  - **Behavior**:
    - If **float** (e.g., ``0.1``), it represents the fraction of the dataset.
    - If **integer** (e.g., ``1000``), it represents the absolute number of samples.

**eval_size**
  The proportion of the training set to use for validation during training.
  - **Type**: ``float``, ``float``
  - **Behavior**:
    - If **float** (e.g., ``0.1``), it represents the fraction of the dataset.
    - If **integer** (e.g., ``1000``), it represents the absolute number of samples.
.. warning::
  Throws an error if the number of points in the eval set is too great (<5000) to avoid performance issues.

**n_folds**
  Only for cross-validation. Number of folds to use. Set to ``null`` to disable.
  - **Type**: ``integer``, ``string``, or ``null``
  - **Options**:
    - ``integer`` (e.g., ``10``): Performs K-fold cross-validation with the specified number of folds.
    - ``"premade"``: Assumes data is already split into fold-specific files. Folder/file structure should follow the pattern of `these <https://github.com/hesther/reactiondatabase/tree/main/data_splits>`_ datasets.
    - ``null``: Disables cross-validation.

---

Vocabulary Settings
===================

Controls the generation and management of the tokenizer's vocabulary file.

**generate_vocab**
  If ``True``, the script will generate a new vocabulary from the training data.
  - **Type**: ``boolean``

**from_scratch**
  If ``True``, any existing vocabulary files will be deleted before generating a new one.
  - **Type**: ``boolean``

**sampling**
  If ``True``, the vocabulary will be generated from a random subsample of the dataset instead of the entire dataset. This is faster but may result in a less complete vocabulary.
  - **Type**: ``boolean``

**sample_size**
  The number of data points to use for vocabulary creation if ``sampling`` is enabled.
  - **Type**: ``integer``

**terminate_after_vocab_creation**
  If ``True``, the script will exit after the vocabulary has been successfully created. This is useful if you only want to perform the vocab generation step.
  - **Type**: ``boolean``

**path_to_vocab_folder**
  The absolute path to the folder where vocabulary files are stored or will be saved.
  - **Type**: ``string``

**max_vocab_size**
  For BPE, the maximal vocabulary size.
  - **Type**: ``integer``

**min_frequency**
  For BPE, minimal frequency for a merge.
  - **Type**: ``integer``

---

Training Settings
=================

All hyperparameters and settings related to the model architecture and training loop.

**num_train_epochs**
  Total number of training epochs to perform (if not an integer, will perform the decimal part percents of the last epoch before stopping training).
  - **Types**: ``integer``

**num_train_steps**
  If set to a positive number, the total number of training steps to perform. Overrides num_train_epochs. For a finite dataset, training is reiterated through the dataset (if all data is exhausted) until max_steps is reached.
  - **Types**: ``integer``

**learning_rate**
  The initial learning rate for AdamW optimizer.
  - **Type**: ``float``

**lr_scheduler_type**
  Learning rate scheduler type used by the Hugging Face Trainer.
  - **Type**: ``string``
  - **Options**: ``linear``, ``cosine``, ``cosine_with_restarts``, ``polynomial``, ``constant``, ``constant_with_warmup``

**hidden_dropout**
  The dropout probability for all fully connected layers in the model.
  - **Type**: ``float``

**model_size**
  A shorthand to set multiple model architecture parameters at once.
  - **Type**: ``string``
  - **Options**: ``"small"``, ``"base"``. The specific dimensions (e.g., ``hidden_size``, ``num_attention_heads``) are defined internally for each model type (BERT/ELECTRA).

**n_layers**
  Used only in **fine-tuning** mode (`regression`/`classification`) to define the feed-forward network (FFN) head placed on top of the transformer.
  The number of layers in the FFN head.
  - **Types**: ``integer``

**layer_width**
  Used only in **fine-tuning** mode (`regression`/`classification`) to define the feed-forward network (FFN) head placed on top of the transformer.
  The width of each layer.
  - **Types**: ``integer``

**mask_p**
  The probability of masking tokens during pre-training for MLM.
  - **Type**: ``float``
.. note::
  This feature is currently deactivated and would require reimplementation.

**max_length**
  The maximum sequence length. Inputs longer than this will be truncated.
  - **Type**: ``integer``

**model_type**
  The base pre-training method to use. ``"bert"`` would be MLM.
  - **Type**: ``string``
  - **Options**: ``"bert"``, ``"electra"``

**model_mode**
  The training objective.
  - **Type**: ``string``
  - **Options**:
    - ``"pretraining"``: For Masked Language Model training.
    - ``"regression"``: For fine-tuning on a continuous target variable.
    - ``"classification"``: For fine-tuning on a categorical target variable.

**run_name**
  Specifies a previous run to load a model from. Can be used for pre-training. Has to be used for fine-tuning.
  - **Type**: ``string`` or ``null``
  - **Behavior**:
    - **In pre-training**: The name of the run to resume training from. Use with ``checkpoint_number``.
    - **In fine-tuning**: The name of the pre-trained model run to use as a starting point.
    - ``"latest"``: Automatically selects the most recently created model run that matches the current ``model_type``, ``model_size`` and ``tokenizer_kind``.

**warmup_ratio**
  The proportion of total training steps used for a linear learning rate warmup.
  - **Type**: ``float``

**gradient_accumulation_steps**
  Number of update steps to accumulate gradients over before performing a backward/update pass. Effectively increases batch size without increasing memory usage.
  - **Type**: ``integer``
.. warning::
  When using gradient accumulation, one step is counted as one step with backward pass. Therefore, logging, evaluation, save will be conducted every gradient_accumulation_steps * xxx_step training examples.

**batch_size**
  The number of samples per device per forward pass.
  - **Type**: ``integer``

**auto_find_batch_size**
  If ``True``, the trainer will attempt to automatically find the largest batch size that fits in memory, starting from the value set in ``batch_size`` and decreasing if an out-of-memory error occurs.
  - **Type**: ``boolean``
  - **Default**: ``False``

**fp16**
  If ``True``, enables mixed-precision training, which can speed up training and reduce memory usage on compatible GPUs.
  - **Type**: ``boolean``

**eval_steps**
  The frequency (in number of training steps) at which to perform evaluation.
  - **Type**: ``integer``

**logging_steps**
  The frequency (in number of training steps) at which to perform logging.
  - **Type**: ``integer``

**early_stopping_patience**
  Parameter to control early stopping and prevent overfitting. The number of evaluation steps to wait for improvement before stopping. ``null`` disables it.
  - **Type**: ``integer`` or ``null``

**early_stopping_threshold**
  Parameters to control early stopping and prevent overfitting. The number of evaluation steps to wait for improvement before stopping. ``null`` disables it.
  - **Type**: ``float``

**checkpoint_number**
  Specifies which checkpoint to load from the ``run_name`` folder.
  - **Type**: ``integer``, ``string``, or ``null``
  - **Options**:
    - ``integer``: The specific checkpoint number (e.g., ``500``).
    - ``"last"``: Loads the most recent checkpoint in the folder. Can be useful if your run crashes.
    - ``null``: Starts training from scratch (or from the base pre-trained model in fine-tuning).

**save_total_limit**
  The maximum number of checkpoints to keep. Older checkpoints are automatically deleted.
  - **Type**: ``integer`` or ``null``
  - **Default**: ``3``
  - **Behavior**: Set to ``null`` to disable this limit and keep all checkpoints.

---

Dataset Settings
================

Controls data loading, preprocessing, and transformations.

**dataset_folder**
  The absolute path to the folder containing the data.
  - **Type**: ``string``

**dataset_name**
  The name of the ``.csv`` file within the dataset folder. For cross-validation, leave the ``.csv`` even if the splits are already premade and in separate folders.
  - **Type**: ``string``

**column_name**
  The name of the column containing the primary input data (e.g., SMILES or CGR strings).
  - **Type**: ``string``

**convert_to_cgr**
  If ``True``, input SMILES strings will be converted into Condensed Graph of Reaction (CGR) format and then to the string-based representation, SMILES/CGR.
  - **Type**: ``boolean``

**label_column_name**
  The name of the column containing the target variable for regression or classification.
  - **Type**: ``string``

**separator**
  The delimiter used in the ``.csv`` file.
  - **Type**: ``string``
  - **Example**: ``,``

**n_max_bond_changes**
  The maximum number of bond changes allowed in a reaction for SMILES/CGR. Used to filter out multi-step reactions.
  - **Type**: ``integer``
  - **Behavior**: A value of ``0`` disables this filter. Should be disabled if not working with CGRs.

**n_min_bond_changes**
  The minimum number of bond changes required in a reaction for SMILES/CGR. Used to filter out entries with no reaction.
  - **Type**: ``integer``
  - **Behavior**: A value of ``0`` disables this filter. Should be disabled if not working with CGRs.

**scale_target**
  If ``True``, the target variable (for regression tasks only) will be scaled using ``sklearn.preprocessing.StandardScaler``.
  - **Type**: ``boolean``

**additional_features**
  A list of column names containing additional numerical features to be concatenated to the model's output before the final prediction layer.
  - **Type**: ``list`` or ``null``
  - **Example**: ``[ "temperature", "pressure" ]``

**invert_reactions**
  If ``True``, the reactants and products of the reactions will be swapped, label is inverted. The result will be ``products>reactants`` and the label will be ``-label``.
  - **Type**: ``boolean``

**augment**
  Specifies the data augmentation strategy and the number of augmented samples to generate for each original data point. Set to ``null`` to disable augmentation.

  - **Type**: ``string`` or ``null``
  - **Syntax**: The value must be a string formatted as ``"type:count"``, where ``type`` is the augmentation method and ``count`` is the number of variants to create per sample.

  **Supported Types:**

  - **smiles**: Creates randomized, but chemically equivalent, SMILES strings. This is useful for teaching the model rotational and atom-ordering invariance.

    *Example*: To generate 10 augmented SMILES for each reaction:

    .. code-block:: yaml

       augment: "smiles:10"

  - **cgr**: Creates variations of the Condensed Graph of Reaction (CGR), for example by altering the atom order while preserving the chemical transformation.

    *Example*: To generate 5 augmented CGRs for each reaction:

    .. code-block:: yaml

       augment: "cgr:5"

**fix_nitro**
  Corrects improperly represented nitro groups using RDKit.
  - **Type**: ``boolean``

**explicify_hydrogens**
  Makes all hydrogen atoms explicit using CGRTools.
  - **Type**: ``boolean``
 .. note::
    Set this to ``True`` if there are transformations involving Hydrogens such as hydrogen atom transfers, or you are going to have a bad time.

**pattern_to_filter_out**
  Filters data points based on the presence or absence of a specific string pattern.
  - **Type**: ``string`` or ``null``
  - **Syntax**:
    - ``"with:(O)"``: Keep only entries containing ``(O)``.
    - ``"without:(O)"``: Discard all entries containing ``(O)``.

**mapping**
  Controls atom mapping in reactions.
  - **Type**: ``string``, ``list``, or ``null``
  - **Options**: ``"remove"``, ``"remap"``, or a list like ``[ "remove", "remap" ]`` which will, in order, remove the mapping and map the reactions again.

**save_dataset_path**
  Path to save the processed and tokenized dataset as a file. Enables caching to speed up subsequent runs.
  - **Type**: ``string`` or ``null``

**load_dataset_path**
  Path to load a previously saved and processed dataset file, skipping the preprocessing steps.
  - **Type**: ``string`` or ``null``

**n_datapoints**
  The number or fraction of rows to use from the dataset.
  - **Type**: ``integer``, ``float``, or ``null``
  - **Behavior**:
    - **integer**: Use the first N rows.
    - **float** (for premade splits): Use a fraction of the training data.
    - **null**: Use the entire dataset.

---

Descriptors Settings
====================

Configuration for including atomic or molecular descriptors as binned features in the input string.

**atomic_descriptors**
  A list of atomic descriptor names to be included as features. Set to ``null`` to disable.
  - **Type**: ``list`` or ``null``
  - **Example**: ``[ "npa_charge", "shielding_constants" ]``

**molecular_descriptors**
  A list of molecular descriptor names to be included as features. This is different from the previous ``additional_descriptors`` as here, the descriptors are included as strings.  Set to ``null`` to disable.
  - **Type**: ``list`` or ``null``
  - **Example**: ``[ "HOMO/LUMO", "IP", "EA" ]``

**descriptors_to_bin**
  A list of numerical descriptors that should be discretized into bins instead of being used as raw float values.
  - **Type**: ``list`` or ``null``

**num_bins**
  The number of discrete bins to create for each descriptor listed in ``descriptors_to_bin``.
  - **Type**: ``integer``

**binning_strategy**
  The method used to create the bins.
  - **Type**: ``string``
  - **Options**: ``'uniform'``, ``'quantile'``, ``'kmeans'``.