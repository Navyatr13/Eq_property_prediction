import optuna
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

# Dummy Model
class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.mse_loss(preds, y)
        self.log("val_loss", loss)
        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

# Dummy Data
train_data = torch.randn(100, 10), torch.randn(100, 1)
val_data = torch.randn(20, 10), torch.randn(20, 1)
train_loader = DataLoader(TensorDataset(*train_data), batch_size=32)
val_loader = DataLoader(TensorDataset(*val_data), batch_size=32)

# Optuna Objective
def objective(trial):
    model = SimpleModel()
    trainer = Trainer(max_epochs=10, enable_checkpointing=False)
    trainer.fit(model, train_loader, val_loader)

    # Manually fetch the best validation loss
    val_loss = trainer.callback_metrics["val_loss"].item()
    trial.report(val_loss, step=0)  # Report the result
    return val_loss

# Optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)
