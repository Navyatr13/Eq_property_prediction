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
# Paths to train and validation directories
train_dir = "./data/processed_train/"
val_dir = "./data/processed_val/"


# Set seed
SEED = 42
set_global_seed(SEED)
# Load data
train_loader, val_loader = get_train_val_loaders(train_dir, val_dir, train_batch_size= 256, val_batch_size=512)

# Define model
model = GNNModel(input_dim=49, hidden_dim=512, output_dim=1, learning_rate= 8.52E-05, dropout_rate = 0.15, weight_decay = 0.002299627)
#model = MPNNModel(input_dim = 6, edge_dim = 3, hidden_dim = 128, output_dim = 2, class_weights = class_weights, learning_rate = 0.001, dropout_rate = 0.36)
print(model)
early_stopping_callback = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=10,         # Stop if no improvement in 10 epochs
        verbose=False,        # Log early stopping events
        mode="min",          # Minimize validation loss
    )
# Train mode
trainer = Trainer(
    max_epochs= 100,
    callbacks=early_stopping_callback,
    gradient_clip_val=1.0,
    devices=1,  # Always set devices to at least 1
    accelerator="gpu" if torch.cuda.is_available() else "cpu"  # Use "gpu" if available, otherwise "cpu"
)

trainer.fit(model, train_loader, val_loader)
plot_losses(model)
plot_roc_curve(model)