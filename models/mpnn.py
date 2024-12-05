import torch
import torch.nn.functional as F

from pytorch_lightning import LightningModule
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, auc
from torch_geometric.nn import GlobalAttention
from torch.nn import Dropout, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import MessagePassing
from sklearn.metrics import roc_auc_score


class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, aggr="add"):
        super(MPNNLayer, self).__init__(aggr=aggr)  # Aggregation: 'add', 'mean', 'max'
        self.edge_embedding = torch.nn.Linear(edge_dim, out_channels)
        self.node_embedding = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Apply edge embedding
        edge_attr = self.edge_embedding(edge_attr)
        x = self.node_embedding(x)
        # Start message passing
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # Combine node features (x_j) and edge features
        return x_j + edge_attr

    def update(self, aggr_out):
        # Update node features (can apply non-linearity here)
        return F.relu(aggr_out)



class MPNNModel(LightningModule):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, class_weights, learning_rate, dropout_rate):
        super(MPNNModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.class_weights = class_weights
        self.dropout_rate = dropout_rate
        self.train_losses = []
        self.train_outputs = []
        self.val_losses = []
        self.validation_outputs = []

        # Define MPNN layers
        self.conv1 = MPNNLayer(input_dim, hidden_dim, edge_dim)
        self.conv2 = MPNNLayer(hidden_dim, hidden_dim, edge_dim)
        
        # Fully connected layer for final output
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = Dropout(dropout_rate)

    def forward(self, batch):
        x, edge_index, edge_attr, batch_indices = batch.x, batch.edge_index, batch.edge_attr, batch.batch

        # Apply MPNN layers
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        # Global pooling for graph-level representation
        x = global_mean_pool(x, batch_indices)

        # Fully connected output layer
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        # Training step
        out = self(batch)  # Forward pass
        target = batch.y.long()
        loss = F.cross_entropy(out, target, weight=self.class_weights)  # Calculate loss
        
        self.train_losses.append(loss.item())
        self.train_outputs.append((out.softmax(dim=1)[:, 1].cpu(), target.detach().cpu()))
        
        self.log('train_loss', loss, prog_bar=True)  # Log training loss
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        out = self(batch)  # Forward pass
        target = batch.y.long()
        loss = F.cross_entropy(out, target, weight=self.class_weights)  # Calculate loss
        acc = (out.argmax(dim=1) == batch.y).float().mean()  # Calculate accuracy
        roc_auc = self.compute_roc_auc(out, target)

        self.val_losses.append(loss.item())
        self.validation_outputs.append((out.softmax(dim=1)[:, 1].cpu(), target.detach().cpu()))
        
        self.log('val_loss', loss, prog_bar=True)  # Log validation loss
        self.log('val_acc', acc, prog_bar=True)  # Log validation accuracy
        self.log('val_roc_auc', roc_auc, prog_bar=False)
        return {"val_loss": loss, "val_roc_auc": roc_auc}
        
    def on_train_epoch_end(self):        
        all_preds, all_targets = zip(*self.train_outputs)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
    
        # Compute ROC-AUC and log it
        fpr, tpr, _ = roc_curve(all_targets.detach().numpy(), all_preds.detach().numpy())
        roc_auc = auc(fpr, tpr)
        self.log('train_roc_auc', roc_auc, prog_bar=True)
    
        # Clear outputs after processing
        self.train_outputs = []

    def on_validation_epoch_end(self):
        # Use the stored predictions for custom metrics
        all_preds, all_targets = zip(*self.validation_outputs)
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
    
        # If not already initialized, create the storage for post-training analysis
        if not hasattr(self, 'post_training_validation_outputs'):
            self.post_training_validation_outputs = (torch.tensor([], device=self.device), 
                                                     torch.tensor([], device=self.device))
        
        # Append current epoch's predictions and targets to the post-training outputs
        self.post_training_validation_outputs = (
            torch.cat([self.post_training_validation_outputs[0], all_preds]),
            torch.cat([self.post_training_validation_outputs[1], all_targets])
        )
    
        # Compute ROC-AUC and log it
        fpr, tpr, _ = roc_curve(all_targets.cpu().detach().numpy(), all_preds.cpu().detach().numpy())
        roc_auc = auc(fpr, tpr)
        self.log('val_roc_auc', roc_auc, prog_bar=True)
    
        # Clear outputs after processing
        self.validation_outputs = []



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay= 0.00075)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def compute_roc_auc(self, predictions, targets):
        # Convert logits to probabilities if necessary
        if predictions.shape[1] > 1:
            probabilities = predictions.softmax(dim=1)[:, 1].detach().cpu().numpy()
        else:
            probabilities = predictions.detach().cpu().numpy()
        # Convert targets to numpy
        targets = targets.detach().cpu().numpy()
        try:
            roc_auc = roc_auc_score(targets, probabilities)
        except ValueError:
            # Handle cases where the AUC score cannot be computed
            roc_auc = 0.0
        return roc_auc

