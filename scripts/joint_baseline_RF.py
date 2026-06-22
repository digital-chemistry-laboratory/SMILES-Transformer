# %%
import drfp
#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import os
import pandas as pd
from smiles_transformer.preprocessing.transform.removeatommappingtransform import RemoveAtomMappingTransform
import numpy as np
import optuna
from sklearn.utils.parallel import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from morgandiff import MorganDifferenceEncoder

# %%
print("starting v2 DRFP RF", flush=True)
base_path="/cluster/project/jorner/gsulpizio/SMILES-Transformer/data/splits/"
#dataset_names=[("snar_processed", "exp_activation_energy"), ("e2sn2", "act"), ("GDB_small_val", "dE0"), ("lograte", "lograte"), ("phosphatase_processed", "Conversion"), ("rad6re", "dh")]
dataset_names=[("rad6re", "dh")]
smiles_col="AAM"
name_output_file="RF_drfp.csv"
print("File will be saved to:", name_output_file, flush=True)
n_jobs_trees=8
data={}
chosen_model=RandomForestRegressor
encoder_used=drfp.DrfpEncoder()
class FingerprintGenerator:
    def encode(self, smiles, include_hydrogens=False):
        return encoder_used.encode(smiles, include_hydrogens=include_hydrogens)
    


#encoder_used = MorganDifferenceEncoder(n_bits=2048, radius=3)

removeatommappingtransform = RemoveAtomMappingTransform()
for dataset_name, target_col in dataset_names:
    data[dataset_name] = {"folds":{}}
    for folder in os.listdir(base_path+dataset_name):
        fold_path=base_path+dataset_name+"/"+folder+"/"
        fold_train=removeatommappingtransform.transform(pd.read_csv(fold_path+"aam_train.csv"), in_column="AAM", out_column="AAM")
        fold_val=removeatommappingtransform.transform(pd.read_csv(fold_path+"aam_val.csv"), in_column="AAM", out_column="AAM")
        fold_test=removeatommappingtransform.transform(pd.read_csv(fold_path+"aam_test.csv"), in_column="AAM", out_column="AAM")
        data[dataset_name]["folds"][folder] = {"train": fold_train, "val": fold_val, "test": fold_test}
        data[dataset_name]["target_col"] = target_col

        


        add_cols=list(data[dataset_name]["folds"][folder]["train"].columns)
        add_cols.remove("AAM")
        try:
            add_cols.remove(target_col)
        except:
            pass
        data[dataset_name]["add_cols"] = add_cols

# %%
if "snar_processed" in data:
    fold_dict=data["snar_processed"]["folds"]
    for fold in fold_dict:
        for item in fold_dict[fold]["test"]["exp_activation_energy"]:
            if not isinstance(item, float):
                print(item)

# %%
class EarlyStoppingCallback:
    """Stop the study if `patience` consecutive completed trials show no improvement."""
    def __init__(self, patience: int, direction: str = "minimize"):
        self.patience = patience
        self.no_improvement_count = 0
        self.best_value = float("inf") if direction == "minimize" else float("-inf")
        self.is_better = (lambda new, old: new < old) if direction == "minimize" \
                         else (lambda new, old: new > old)

    def __call__(self, study, trial):
        # Only completed trials count. Pruned/failed trials are skipped.
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return

        if self.is_better(study.best_value, self.best_value):
            self.best_value = study.best_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            print(f"Early stopping: no improvement in {self.patience} completed trials. "
                  f"Best value: {self.best_value:.6f}", flush=True)
            study.stop()
            
# %%
def optimize_xgboost(folds, n_trials=50):
    """
    Optimizes RF hyperparameters using Optuna jointly over all folds of a dataset.
    The objective is the mean validation RMSE across folds, so a single set of
    hyperparameters is selected for the whole dataset.
    """

    def objective(trial):
        n_estimators_max = trial.suggest_int('n_estimators', 100, 1000)
        step = 64  # add (step) trees per pruning checkpoint

        params = {
            'max_depth':         trial.suggest_int('max_depth', 5, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'bootstrap':         trial.suggest_categorical('bootstrap', [True, False]),
            'n_jobs':            n_jobs_trees,
            'random_state':      42,
            'warm_start':        True,
            'n_estimators':      step,
        }

        # One warm-started model per fold, grown in lockstep.
        models = [chosen_model(**params) for _ in folds]

        rmse = None
        for n_trees in range(step, n_estimators_max + 1, step):
            fold_rmses = []
            for model, f in zip(models, folds):
                model.n_estimators = n_trees
                model.fit(f["X_train"], f["y_train"])     # adds (step) new trees
                fold_rmses.append(root_mean_squared_error(f["y_val"], model.predict(f["X_val"])))
            rmse = float(np.mean(fold_rmses))

            trial.report(rmse, step=n_trees)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return rmse

    # 1. Create study and optimize
    print(f"Starting Optuna optimization for {n_trials} trials...")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=150),)
    study.optimize(objective, n_trials=n_trials, callbacks=[EarlyStoppingCallback(patience=10)], n_jobs=os.cpu_count()//n_jobs_trees ,
)

    # 2. Extract best parameters (shared across all folds of this dataset)
    best_params = study.best_params
    best_params.update({'n_jobs': -1, 'random_state': 42})

    print(f"Trials: {len(study.trials)} | "
      f"Completed: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)} | "
      f"Pruned: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}",
      flush=True)

    return best_params


def evaluate_fold(best_params, fold):
    """Train a final model on the fold's train set with the shared best params
    and return test metrics for that fold."""
    final_model = chosen_model(**best_params)
    final_model.fit(fold["X_train"], fold["y_train"])

    test_preds = final_model.predict(fold["X_test"])
    test_mse = mean_squared_error(fold["y_test"], test_preds)
    test_mae = mean_absolute_error(fold["y_test"], test_preds)
    test_r2 = r2_score(fold["y_test"], test_preds)

    return test_mse, test_mae, test_r2, test_preds

# %%


def _encode_chunk(chunk, include_hydrogens=False):
    """Encode a list of SMILES; on batch failure, fall back per-SMILES."""
    try:
        return encoder_used.encode(chunk, include_hydrogens=include_hydrogens), [True] * len(chunk)
    except Exception:
        fps, mask = [], []
        for s in chunk:
            try:
                fps.append(encoder_used.encode([s], include_hydrogens=include_hydrogens)[0])
                mask.append(True)
            except Exception:
                mask.append(False)
        return fps, mask

def safe_encode(df, smiles_col, n_jobs=-1, chunk_size=256, include_hydrogens=False):
    smiles_list = df[smiles_col].tolist()
    chunks = [smiles_list[i:i+chunk_size]
              for i in range(0, len(smiles_list), chunk_size)]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_encode_chunk)(c, include_hydrogens=include_hydrogens) for c in chunks
    )

    fps_all, mask_all = [], []
    for fps, mask in results:
        fps_all.extend(fps)
        mask_all.extend(mask)

    mask_arr = np.array(mask_all, dtype=bool)
    filtered_df = df[mask_arr].copy()
    X = np.vstack(fps_all) if fps_all else np.array([])
    err = int((~mask_arr).sum())
    rate = err / len(df) if len(df) else 0.0
    return filtered_df, X, err, rate

# %%
# Dictionary to store the error tracking for reporting



reaction_error_tracking = {}
results_list = []

for dataset in data:
    reaction_error_tracking[dataset] = {}
    folds_data = []  # encoded data for every fold of this dataset

    for fold in data[dataset]["folds"]:
        # 1. Fetch raw DataFrames for the current fold
        df_train_raw = data[dataset]["folds"][fold]["train"]
        df_val_raw   = data[dataset]["folds"][fold]["val"]
        df_test_raw  = data[dataset]["folds"][fold]["test"]

        # 2. Safely encode and filter the DataFrames
        df_train, X_train, tr_err, tr_rate = safe_encode(df_train_raw, smiles_col, include_hydrogens=True if dataset == "rad6re" else False)
        df_val, X_val, val_err, val_rate   = safe_encode(df_val_raw, smiles_col, include_hydrogens=True if dataset == "rad6re" else False)
        df_test, X_test, te_err, te_rate   = safe_encode(df_test_raw, smiles_col, include_hydrogens=True if dataset == "rad6re" else False)

        # Log the error rates for this fold
        reaction_error_tracking[dataset][fold] = {
            "train_error_rate": tr_rate,
            "val_error_rate": val_rate,
            "test_error_rate": te_rate,
            "train_dropped": tr_err,
            "val_dropped": val_err,
            "test_dropped": te_err
        }

        # 3. Handle additional columns using the FILTERED DataFrames
        if data[dataset]["add_cols"]:
            # Ensure we are using the filtered df_train to get column names and data
            add_cols = data[dataset]["add_cols"]
            X_train = np.hstack((X_train, df_train[add_cols].to_numpy()))
            X_val   = np.hstack((X_val, df_val[add_cols].to_numpy()))
            X_test  = np.hstack((X_test, df_test[add_cols].to_numpy()))

        print(f"Dataset: {dataset} | Fold: {fold}", flush=True)
        print(f"Shapes (Train, Val, Test): {X_train.shape}, {X_val.shape}, {X_test.shape}", flush=True)
        print(f"Dropped rows (Train, Val, Test): {tr_err}, {val_err}, {te_err}", flush=True)

        # 4. Extract target values (y) using the FILTERED DataFrames
        y_col_name = data[dataset]["target_col"]
        y_train = df_train[y_col_name].to_numpy()
        y_val   = df_val[y_col_name].to_numpy()
        y_test  = df_test[y_col_name].to_numpy()

        folds_data.append({
            "fold": fold,
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
            "train_error_rate": tr_rate,
            "test_error_rate": te_rate,
        })

    # 5. Optimize hyperparameters jointly over all folds -> one param set per dataset
    best_params = optimize_xgboost(folds_data, n_trials=50)

    # 6. Train per fold with the shared best params and report test metrics per fold
    for f in folds_data:
        test_mse, test_mae, test_r2, test_preds = evaluate_fold(best_params, f)

        results_list.append({
            "dataset": dataset,
            "fold": f["fold"],
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_r2": test_r2,
            "best_params": best_params,
            "train_error_rate": f["train_error_rate"],
            "test_error_rate": f["test_error_rate"]
        })

# Convert final results to a DataFrame
results_df = pd.DataFrame(results_list)

# %%
results_df.to_csv(name_output_file, index=False)

