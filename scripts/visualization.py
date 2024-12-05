import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch

def visualize_embeddings(embeddings_dict, targets_dict, method="t-SNE"):
    plt.figure(figsize=(12, 8))
    for epoch, embeddings in embeddings_dict.items():
        targets = targets_dict[epoch]
        
        # Dimensionality reduction
        if method == "t-SNE":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "UMAP":
            reducer = UMAP(n_components=2, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=targets, cmap='coolwarm', alpha=0.6)
        plt.title(f"{method} Visualization - Epoch {epoch}")
        plt.colorbar(label="Class")
        plt.show()
    
def plot_losses(model):
    plt.figure(figsize=(10, 6))
    plt.plot(model.train_losses, label='Training Loss')
    plt.plot(model.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.show()

def plot_roc_curve(model):
    if not hasattr(model, 'post_training_validation_outputs') or model.post_training_validation_outputs is None:
        print("No validation outputs to plot.")
        return

    preds, targets = model.post_training_validation_outputs
    preds = preds.cpu().numpy()  # Convert to NumPy for compatibility with sklearn
    targets = targets.cpu().numpy()

    # Compute and plot ROC curve
    fpr, tpr, _ = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.show()


