import os
import torch
from torch_geometric.data import DataLoader

def process_batches(batch_dir, batch_size=32, shuffle=True):
    """
    Create DataLoader for batches of graphs directly from disk.

    Args:
        batch_dir (str): Directory containing batch files.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: A DataLoader object containing the loaded batches.
    """
    batch_files = sorted([os.path.join(batch_dir, f) for f in os.listdir(batch_dir) if f.endswith(".pt")])
    print(f"Found {len(batch_files)} batch files in {batch_dir}.")

    all_graphs = []
    for batch_idx, batch_file in enumerate(batch_files):
        # Load the batch from disk
        batch_graphs = torch.load(batch_file)
        for i, graph in enumerate(batch_graphs):
            if graph is None:
                print(f"Graph at index {i} in batch {batch_idx} is None")
        #print(f"{batch_file} contains {len(batch_graphs)} graphs.")
        all_graphs.extend(batch_graphs)  # Combine all graphs from the batch files

    # Create and return DataLoader
    loader = DataLoader(all_graphs, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_train_val_loaders(train_dir, val_dir, test_dir, train_batch_size=32, val_batch_size=32, test_batch_size = 32):
    """
    Generate DataLoaders for both training and validation data.

    Args:
        train_dir (str): Directory containing training batches.
        val_dir (str): Directory containing validation batches.
        train_batch_size (int): Batch size for training DataLoader.
        val_batch_size (int): Batch size for validation DataLoader.

    Returns:
        tuple: (train_loader, val_loader)
    """
    train_loader = process_batches(train_dir, batch_size=train_batch_size, shuffle=True)
    val_loader = process_batches(val_dir, batch_size=val_batch_size, shuffle=False)
    test_loader = process_batches(test_dir, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# usage
train_dir = "./data/processed_train/"
val_dir = "./data/processed_val/"
test_dir = "./data/processed_test/"

#train_loader, val_loader, test_loader = get_train_val_loaders(train_dir, val_dir, test_dir)
