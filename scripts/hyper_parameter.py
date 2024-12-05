import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer
from data_processing.data_loading import get_train_val_loaders
from models.gnn import GNNModel
import torch
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def objective(trial):
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256,512])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    train_batch_size = trial.suggest_categorical("train_batch_size", [64, 128, 256, 512, 1024])
    val_batch_size = train_batch_size  # Keep validation batch size the same

    # Load data
    train_loader, val_loader = get_train_val_loaders(
        train_dir="D:/Equivarance_property_prediction/data/processed_train/",
        val_dir="D:/Equivarance_property_prediction/data/processed_val/",
        train_batch_size=train_batch_size,
        val_batch_size= train_batch_size,
    )
    
    #class_weights = torch.tensor([0.12181257, 0.8781874])

    # Define the model
    model = GNNModel(
        input_dim=6,
        hidden_dim=hidden_dim,
        output_dim=1,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_decay = weight_decay
    )

    # Define the trainer
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_roc_auc")
    early_stopping_callback = EarlyStopping(
        monitor="val_roc_auc",  # Monitor validation loss
        patience=10,         # Stop if no improvement in 10 epochs
        verbose=False,        # Log early stopping events
        mode="min",          # Minimize validation loss
    )
    trainer = Trainer(
        max_epochs=50,
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[pruning_callback, early_stopping_callback],
        enable_progress_bar=True,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    # Return the validation ROC-AUC
    val_loss = trainer.callback_metrics.get("val_roc_auc")
    if val_loss is None:
        raise ValueError("Validation ROC-AUC not found. Ensure it's being logged.")
    return val_loss.item()


# Run the Optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print("Best hyperparameters:")
print(study.best_params)

# Optionally save the study
study.trials_dataframe().to_csv("optuna_trials_new.csv")
