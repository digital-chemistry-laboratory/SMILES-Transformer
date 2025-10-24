# In this file, define the hyperparameters you want to search for.
def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-6, 1e-4, log=True
        ),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [16, 32, 64, 128]
        ),
    }
