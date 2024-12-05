import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP 
from pytorch_lightning import Trainer
from data_processing.data_loading import get_train_val_loaders
from models.gnn import GNNModel
from models.mpnn import MPNNModel
from visualization import plot_losses, plot_roc_curve
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import set_global_seed
from config import *

# Paths to train and validation directories
train_dir = DATASET_PATHS["train"]["output"]
val_dir = DATASET_PATHS["val"]["output"]
test_dir = DATASET_PATHS["test"]["output"]

# Set seed
SEED = 42
set_global_seed(SEED)
# Load data
train_loader, val_loader, test_loader = get_train_val_loaders(
    train_dir,
    val_dir,
    test_dir,
    train_batch_size=TRAINING_CONFIG["train_batch_size"],
    val_batch_size=TRAINING_CONFIG["val_batch_size"],
    test_batch_size=TRAINING_CONFIG["test_batch_size"]
)
# Define model
selected_model = "GNNModel"
MODEL_CONFIG = MODELS[selected_model]

model = GNNModel(
    input_dim=MODEL_CONFIG["input_dim"],
    hidden_dim=MODEL_CONFIG["hidden_dim"],
    output_dim=MODEL_CONFIG["output_dim"],
    learning_rate=MODEL_CONFIG["learning_rate"],
    dropout_rate=MODEL_CONFIG["dropout_rate"],
    weight_decay=MODEL_CONFIG["weight_decay"]
)
print(model)
# Define early stopping
early_stopping_callback = EarlyStopping(
    monitor=TRAINING_CONFIG["monitor_metric"],
    patience=TRAINING_CONFIG["early_stopping_patience"],
    verbose=False,
    mode=TRAINING_CONFIG["monitor_mode"]
)
# Train mode
trainer = Trainer(
    max_epochs=TRAINING_CONFIG["max_epochs"],
    callbacks=early_stopping_callback,
    gradient_clip_val=TRAINING_CONFIG["gradient_clip_val"],
    devices=1,
    accelerator="gpu" if torch.cuda.is_available() else "cpu"
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)

plot_losses(model)
plot_roc_curve(model)