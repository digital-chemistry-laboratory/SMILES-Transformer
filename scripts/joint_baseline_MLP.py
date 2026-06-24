# %%
import drfp
#from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
import os
import pandas as pd
from smiles_transformer.preprocessing.transform import RemoveAtomMappingTransform, ExplicifyHydrogensTransform
import numpy as np
import optuna
from sklearn.utils.parallel import Parallel, delayed
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from morgandiff import MorganDifferenceEncoder
# %%
print("starting v2 MLP morgan diff", flush=True)
base_path="/cluster/project/jorner/gsulpizio/SMILES-Transformer/data/splits/"
dataset_names=[("snar_processed", "exp_activation_energy"), ("e2sn2", "act"), ("GDB_small_val", "dE0"), ("lograte", "lograte"), ("phosphatase_processed", "Conversion"), ("rad6re", "dh")]
#dataset_names=[("rad6re", "dh")]
smiles_col="AAM"
name_output_file= "MLP_drfp.csv"
print("File will be saved to:", name_output_file, flush=True)
n_jobs_trees=8
data={}

base_drfp_encoder = drfp.DrfpEncoder() #MorganDifferenceEncoder(n_bits=2048, radius=3)
class FingerprintGenerator:
    def encode(self, smiles, include_hydrogens=False):
        return base_drfp_encoder.encode(smiles, include_hydrogens=include_hydrogens)

encoder_used = FingerprintGenerator()

class NoamLikeLR:
    """Linear warmup from init_lr to max_lr over `warmup_steps`,
    then exponential decay to final_lr over the remaining steps."""
    def __init__(self, optimizer, warmup_steps, total_steps,
                 init_lr=1e-4, max_lr=1e-3, final_lr=1e-4):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.decay_steps  = max(1, total_steps - self.warmup_steps)
        self.init_lr, self.max_lr, self.final_lr = init_lr, max_lr, final_lr
        self.gamma = (final_lr / max_lr) ** (1.0 / self.decay_steps)
        self.step_num = 0
        self._set(init_lr)

    def _set(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            frac = self.step_num / self.warmup_steps
            lr = self.init_lr + frac * (self.max_lr - self.init_lr)
        else:
            lr = self.max_lr * (self.gamma ** (self.step_num - self.warmup_steps))
        self._set(lr)
        
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, ffn_layers, dropout=0.0):
        super().__init__()
        layers = [nn.Dropout(dropout)] if dropout > 0 else []
        if ffn_layers == 1:
            layers.append(nn.Linear(input_dim, 1))
        else:
            layers += [nn.Linear(input_dim, hidden_size), nn.ReLU()]
            for _ in range(ffn_layers - 2):
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epochs(model, loader, optimizer, loss_fn, n_epochs, device):
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def predict_np(model, X, device, batch_size=1024):
    model.eval()
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    out = []
    for i in range(0, len(X_t), batch_size):
        out.append(model(X_t[i:i+batch_size]).cpu().numpy())
    return np.concatenate(out)


remove_transform = RemoveAtomMappingTransform()
explicit_transform = ExplicifyHydrogensTransform(mapping="remove")

for dataset_name, target_col in dataset_names:
    data[dataset_name] = {"folds":{}}
    for folder in os.listdir(base_path+dataset_name):
        fold_path=base_path+dataset_name+"/"+folder+"/"

        fold_train=remove_transform.transform(pd.read_csv(fold_path+"aam_train.csv"), in_column=smiles_col, out_column=smiles_col)
        fold_val=remove_transform.transform(pd.read_csv(fold_path+"aam_val.csv"), in_column=smiles_col, out_column=smiles_col)
        fold_test=remove_transform.transform(pd.read_csv(fold_path+"aam_test.csv"), in_column=smiles_col, out_column=smiles_col)
        if dataset_name=="rad6re":
            fold_train=explicit_transform.transform(fold_train, in_column=smiles_col, out_column=smiles_col)
            fold_val=explicit_transform.transform(fold_val, in_column=smiles_col, out_column=smiles_col)
            fold_test=explicit_transform.transform(fold_test, in_column=smiles_col, out_column=smiles_col)

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE     = 50      # Chemprop default
WARMUP_EPOCHS  = 2
REPORT_EVERY   = 2       # epochs between Optuna pruning checks

def _make_optimizer(model, warmup_steps, total_steps):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = NoamLikeLR(optimizer, warmup_steps, total_steps,
                           init_lr=1e-4, max_lr=1e-3, final_lr=1e-4)
    loss_fn = nn.MSELoss()
    return optimizer, scheduler, loss_fn


def optimize_mlp(folds, n_trials=50, n_epochs=100):
    """Optimize MLP hyperparameters jointly over all folds of a dataset.
    The objective is the mean validation RMSE across folds, so a single set of
    hyperparameters is selected for the whole dataset."""

    # Pre-scale each fold once (scalers are fit on each fold's own train set).
    prepared = []
    for f in folds:
        x_scaler = StandardScaler().fit(f["X_train"])
        X_train_s = x_scaler.transform(f["X_train"]).astype(np.float32)
        X_val_s   = x_scaler.transform(f["X_val"]).astype(np.float32)

        y_scaler = StandardScaler().fit(f["y_train"].reshape(-1, 1))
        y_train_s = y_scaler.transform(f["y_train"].reshape(-1, 1)).ravel().astype(np.float32)

        Xt_train = torch.as_tensor(X_train_s)
        yt_train = torch.as_tensor(y_train_s)
        n_batches = max(1, (len(Xt_train) + BATCH_SIZE - 1) // BATCH_SIZE)

        prepared.append({
            "Xt_train": Xt_train, "yt_train": yt_train,
            "X_val_s": X_val_s, "y_val": f["y_val"], "y_scaler": y_scaler,
            "input_dim": X_train_s.shape[1],
            "total_steps": n_epochs * n_batches,
            "warmup_steps": WARMUP_EPOCHS * n_batches,
        })

    def objective(trial):
        hidden_size = trial.suggest_int('hidden_size', 300, 2400, step=100)
        ffn_layers  = trial.suggest_int('ffn_layers', 1, 3)
        dropout     = trial.suggest_float('dropout', 0.0, 0.4)

        # One model per fold, trained in lockstep epoch by epoch.
        states = []
        for p in prepared:
            torch.manual_seed(42)
            model = MLP(p["input_dim"], hidden_size, ffn_layers, dropout).to(device)
            optimizer, scheduler, loss_fn = _make_optimizer(
                model, p["warmup_steps"], p["total_steps"])
            loader = DataLoader(TensorDataset(p["Xt_train"], p["yt_train"]),
                                batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
            states.append((model, optimizer, scheduler, loss_fn, loader))

        rmse = None
        for epoch in range(1, n_epochs + 1):
            for model, optimizer, scheduler, loss_fn, loader in states:
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    loss_fn(model(xb), yb).backward()
                    optimizer.step()
                    scheduler.step()

            if epoch % REPORT_EVERY == 0 or epoch == n_epochs:
                fold_rmses = []
                for (model, *_), p in zip(states, prepared):
                    preds_val_s = predict_np(model, p["X_val_s"], device)
                    preds_val   = p["y_scaler"].inverse_transform(preds_val_s.reshape(-1, 1)).ravel()
                    fold_rmses.append(root_mean_squared_error(p["y_val"], preds_val))
                rmse = float(np.mean(fold_rmses))
                trial.report(rmse, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return rmse

    print(f"Starting Optuna optimization for {n_trials} trials...")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=8),
    )
    n_parallel = 1 if device == 'cuda' else max(1, os.cpu_count() // 2)
    if device != 'cuda':
        torch.set_num_threads(2)
    study.optimize(objective, n_trials=n_trials,
                   callbacks=[EarlyStoppingCallback(patience=10)],
                   n_jobs=n_parallel)

    print(f"Trials: {len(study.trials)} | "
      f"Completed: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)} | "
      f"Pruned: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}",
      flush=True)

    return study.best_params


def evaluate_fold(best, fold, n_epochs=100):
    """Train a final MLP on the fold's train set with the shared best params
    and return test metrics for that fold."""
    x_scaler = StandardScaler().fit(fold["X_train"])
    X_train_s = x_scaler.transform(fold["X_train"]).astype(np.float32)
    X_test_s  = x_scaler.transform(fold["X_test"]).astype(np.float32)

    y_scaler = StandardScaler().fit(fold["y_train"].reshape(-1, 1))
    y_train_s = y_scaler.transform(fold["y_train"].reshape(-1, 1)).ravel().astype(np.float32)

    Xt_train = torch.as_tensor(X_train_s)
    yt_train = torch.as_tensor(y_train_s)
    n_batches    = max(1, (len(Xt_train) + BATCH_SIZE - 1) // BATCH_SIZE)
    total_steps  = n_epochs * n_batches
    warmup_steps = WARMUP_EPOCHS * n_batches

    torch.manual_seed(42)
    final_model = MLP(X_train_s.shape[1], best['hidden_size'],
                      best['ffn_layers'], best['dropout']).to(device)
    optimizer, scheduler, loss_fn = _make_optimizer(final_model, warmup_steps, total_steps)
    loader = DataLoader(TensorDataset(Xt_train, yt_train),
                        batch_size=BATCH_SIZE, shuffle=True)
    for _ in range(n_epochs):
        final_model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss_fn(final_model(xb), yb).backward()
            optimizer.step()
            scheduler.step()

    test_preds_s = predict_np(final_model, X_test_s, device)
    test_preds   = y_scaler.inverse_transform(test_preds_s.reshape(-1, 1)).ravel()

    test_mse = mean_squared_error(fold["y_test"], test_preds)
    test_mae = mean_absolute_error(fold["y_test"], test_preds)
    test_r2  = r2_score(fold["y_test"], test_preds)

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

    n_epochs = 500 if dataset == "snar_processed" else 100

    # 5. Optimize hyperparameters jointly over all folds -> one param set per dataset
    best_params = optimize_mlp(folds_data, n_trials=50, n_epochs=n_epochs)

    # 6. Train per fold with the shared best params and report test metrics per fold
    for f in folds_data:
        test_mse, test_mae, test_r2, test_preds = evaluate_fold(best_params, f, n_epochs=n_epochs)

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
