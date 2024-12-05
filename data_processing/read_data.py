import pandas as pd
import os
from torch_geometric.loader import DataLoader
from .mol_to_graph import MoleculeDataset

def read_smiles_data(filepath):
    """
    Reads a CSV file containing SMILES strings and target properties.
    
    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['smiles', 'Toxicity Value'].
    """
    data = pd.read_csv(filepath)
    data = data.sample(frac=1).reset_index(drop=True) 
    if 'Canonical_SMILES' not in data.columns or 'Toxicity_Value' not in data.columns:
        raise ValueError("CSV must contain 'smiles' and 'target' columns.")
    return data
