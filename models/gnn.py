import torch
import torch.nn.functional as F
import numpy as np
import umap
import matplotlib.pyplot as plt

from pytorch_lightning import LightningModule
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, auc
from torch_geometric.nn import GlobalAttention
from torch.nn import Dropout, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score


import torch.nn as nn
import torch.nn.init as init

def initialize_weights(module):
    """
    Applies appropriate weight initialization to the module.
    """
    if isinstance(module, nn.Linear):  # For fully connected layers
        init.xavier_uniform_(module.weight)  # Xavier uniform initialization
        if module.bias is not None:
            init.zeros_(module.bias)  # Initialize biases to zero
    elif isinstance(module, nn.Conv2d):  # For convolutional layers
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):  # For BatchNorm
        init.ones_(module.weight)  # Initialize scale to 1
        init.zeros_(module.bias)  # Initialize shift to 0
    elif hasattr(module, 'reset_parameters'):  # For PyTorch Geometric layers (e.g., GCNConv)
        module.reset_parameters()

class GNNModel(LightningModule):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate,dropout_rate,weight_decay):
        super(GNNModel, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.train_losses = []
        self.train_outputs = []
        self.val_losses = []
        self.validation_outputs = []
        self.validation_embeddings = []
        self.validation_targets = []
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay


        # Define the GNN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim // 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout_rate),
                    torch.nn.Linear(hidden_dim//2, hidden_dim // 4),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout_rate),
                    torch.nn.Linear(hidden_dim // 4, output_dim),
                )

        self.attn_pool = GlobalAttention(torch.nn.Linear(hidden_dim, 1))
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = Dropout(self.dropout_rate)

        self.apply(initialize_weights)

    def forward(self, batch):
        # Forward pass
        x, edge_index, batch_indices, edge_attr = batch.x, batch.edge_index, batch.batch, batch.edge_attr 
        edge_weight = edge_attr#.mean(dim=1) 
        x = self.conv1(x, edge_index)#, edge_weight=edge_weight)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)#, edge_weight=edge_weight)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)#, edge_weight=edge_weight)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        #x = self.attn_pool(x, batch_indices)
        embeddings = global_mean_pool(x, batch_indices)  # Pool over graph nodes
        x = self.mlp(embeddings)
        return x, embeddings

    def training_step(self, batch, batch_idx):
        # Training step
        out,_ = self(batch)
        out = out.squeeze()  # Forward pass
        target = batch.y.float()
        loss = F.binary_cross_entropy_with_logits(out, target)  # Calculate loss
        probabilities = torch.sigmoid(out)
        
        self.train_losses.append(loss.item())
        self.train_outputs.append((probabilities.cpu(), target.cpu()))
        self.log('train_loss', loss, prog_bar=True)  # Log training loss
        return loss

    def validation_step(self, batch, batch_idx):
        print(batch)
        # Validation step
        '''out = self(batch)  # Forward pass
        target = batch.y.long()
        loss = F.cross_entropy(out, target, weight=self.class_weights)  # Calculate loss
        acc = (out.argmax(dim=1) == batch.y).float().mean()  # Calculate accuracy
        roc_auc = self.compute_roc_auc(out, target)'''
        out, embeddings = self(batch)  # Forward pass
        out = out.squeeze()
        target = batch.y.float()
        if not hasattr(self, 'epoch_wise_embeddings'):
            self.epoch_wise_embeddings = {}
            self.epoch_wise_targets = {}
        if not hasattr(self, 'current_epoch'):  # Track epoch manually in Lightning
            self.current_epoch = 0
    
        if self.current_epoch % 10 == 0:  # Save every 10 epochs
            if self.current_epoch not in self.epoch_wise_embeddings:
                self.epoch_wise_embeddings[self.current_epoch] = []
                self.epoch_wise_targets[self.current_epoch] = []
            self.epoch_wise_embeddings[self.current_epoch].append(embeddings)
            self.epoch_wise_targets[self.current_epoch].append(target.detach().cpu())

        # Add to validation embeddings and targets for epoch-wise processing
        self.validation_embeddings.append(embeddings.detach())
        self.validation_targets.append(target.detach())
    
        # Calculate loss
        loss = F.binary_cross_entropy_with_logits(out, target)
        
        # Convert logits to probabilities
        probabilities = torch.sigmoid(out)
        # Calculate accuracy
        predicted = (probabilities > 0.5).float()  # Threshold at 0.5
        accuracy = (predicted == target).float().mean()  # Accuracy calculation
        # Calculate ROC-AUC
        roc_auc = self.compute_roc_auc(probabilities, target)

        self.val_losses.append(loss.item())
        self.validation_outputs.append((probabilities.cpu(), target.detach().cpu()))
        
        self.log('val_loss', loss, prog_bar=True)  # Log validation loss
        self.log('val_acc', accuracy, prog_bar=True)  # Log validation accuracy
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
    
        # Handle embeddings
        embeddings = torch.cat(self.validation_embeddings, dim=0)
        targets = torch.cat(self.validation_targets, dim=0)

        # Reset validation embeddings and targets for the next epoch
        self.validation_embeddings = []
        self.validation_targets = []
        
        # Check if the current epoch is a multiple of 10
        current_epoch = self.current_epoch
        if current_epoch % 10 == 0:
            # Save embeddings and targets for visualization
            self.final_validation_embeddings = embeddings.cpu().detach().numpy()
            self.final_validation_targets = targets.cpu().detach().numpy()
    
            # Optionally visualize embeddings using UMAP
            self.visualize_embeddings(self.final_validation_embeddings, self.final_validation_targets, current_epoch)
    
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay= self.weight_decay )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def compute_roc_auc(self, predictions, targets):
        """
        Computes the ROC-AUC score for binary classification.
        """
        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(predictions).detach().cpu().numpy()
    
        # Convert targets to numpy
        targets = targets.detach().cpu().numpy()
    
        try:
            roc_auc = roc_auc_score(targets, probabilities)
        except ValueError:
            # Handle cases where the AUC score cannot be computed
            roc_auc = 0.0
        return roc_auc

    def compute_roc_auc_ce(self, predictions, targets):
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
    def visualize_embeddings(self, embeddings, targets, epoch):
        """
        Visualize the embeddings using UMAP and save the plot.
        """    
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
    
        plt.figure(figsize=(8, 6))
        for label in np.unique(targets):
            indices = targets == label
            plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1], label=f'Class {label}', alpha=0.7)
        
        plt.title(f"UMAP Visualization of Validation Embeddings (Epoch {epoch})")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.legend()
        plt.savefig(f"validation_embeddings_epoch_{epoch}.png")
        plt.close()
