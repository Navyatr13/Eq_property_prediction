import torch
import os
import sys
from torch_geometric.data import Dataset, Data  # Ensure both Dataset and Data are imported
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logging
import warnings
from rdkit import RDLogger
from sklearn.preprocessing import MinMaxScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
# Suppress all warnings
warnings.filterwarnings("ignore")
# Suppress RDKit-specific warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
scripts_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(scripts_dir, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Configure logging to save logs in the scripts/logs directory
log_file = os.path.join(log_dir, "pipeline.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

unique_atomic_numbers = [1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 44, 46, 48, 50, 53, 56, 60, 78, 79, 80, 81, 83]


def get_node_features(mol, unique_atomic_numbers):
    """
    Generate node features for a molecule, including one-hot encoding for atomic numbers and normalized additional features.

    Args:
        mol: RDKit molecule object.
        unique_atomic_numbers: List of unique atomic numbers in the dataset.

    Returns:
        torch.Tensor: Node feature matrix.
    """
    one_hot_encoded = []
    additional_features = []

    for atom in mol.GetAtoms():
        # Atomic number (categorical - one-hot encoded)
        atomic_number = atom.GetAtomicNum()
        one_hot_encoded.append([atomic_number])

        # Additional features
        additional_features.append([
            int(atom.GetIsAromatic()),          # Aromaticity (0/1)
            atom.GetDegree(),                   # Atom degree (int)
            atom.GetTotalValence(),             # Total valence electrons (int)
            atom.GetFormalCharge(),             # Formal charge (int)
            int(atom.GetHybridization()),       # Hybridization as integer enum
        ])

    # Convert atomic numbers to one-hot encoding
    one_hot_encoded = np.array(one_hot_encoded).reshape(-1, 1)
    encoder = OneHotEncoder(categories=[unique_atomic_numbers], sparse_output=False, handle_unknown="ignore")
    one_hot_encoded = encoder.fit_transform(one_hot_encoded)

    # Normalize additional features
    additional_features = np.array(additional_features, dtype=np.float32)
    scaler = MinMaxScaler()
    normalized_additional_features = scaler.fit_transform(additional_features)

    # Concatenate one-hot encoded atomic numbers with normalized additional features
    node_features = np.hstack((one_hot_encoded, normalized_additional_features))

    return torch.tensor(node_features, dtype=torch.float)


def get_edge_features(mol, conformer):
    """
        get edge featues of each molecule

        Args:
            mol : RDKit molecule object.
            conformer : RDKit molecule object.

        Returns:
            torch.Tensor: edge feature matrix.
        """
    edge_features = []
    edge_indices = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_length = conformer.GetAtomPosition(i).Distance(conformer.GetAtomPosition(j))

        edge_features.append([
            int(bond.GetBondTypeAsDouble()),  # Bond type as double
            bond.GetIsConjugated(),          # Is conjugated (True/False)
            bond_length                      # Bond length
        ])
        # Add both directions (undirected graph)
        edge_indices.append((i, j))
        edge_indices.append((j, i))

    edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features * 2, dtype=torch.float)  # Duplicate for both directions
    return edge_indices, edge_features




def molecule_to_graph(smiles, target):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")

        mol = Chem.AddHs(mol)
        status = AllChem.EmbedMolecule(mol)
        if status != 0:
            raise ValueError(f"Conformer generation failed for SMILES: {smiles}")

        conformer = mol.GetConformer()
        x = get_node_features(mol, unique_atomic_numbers)
        edge_index, edge_attr = get_edge_features(mol, conformer)
        pos = torch.tensor([[conformer.GetAtomPosition(i).x,
                             conformer.GetAtomPosition(i).y,
                             conformer.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())], dtype=torch.float)

        pos_mean = pos.mean(dim=0, keepdim=True)
        pos_centered = pos - pos_mean
        max_dist = pos_centered.norm(p=2, dim=1).max()
        if max_dist == 0:
            max_dist = 1.0
        pos_normalized = pos_centered / max_dist

        x = torch.cat([x, pos_normalized], dim=1)
        y = torch.tensor([target], dtype=torch.float)

        if torch.isnan(x).any() or torch.isnan(edge_attr).any() or torch.isnan(pos_normalized).any():
            raise ValueError("NaN detected in graph components.")

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos_normalized, y=y)
    except Exception as e:
        logging.error(f"Error processing molecule: {smiles}, Error: {e}")
        raise


def get_unique_atomic_numbers(dataset_smiles):
    """
    Identify unique atomic numbers (elements) in the dataset.

    Args:
        dataset_smiles (list[str]): List of SMILES strings in the dataset.

    Returns:
        list[int]: Sorted list of unique atomic numbers.
    """
    unique_atoms = set()
    for smiles in dataset_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            for atom in mol.GetAtoms():
                unique_atoms.add(atom.GetAtomicNum())
    return sorted(unique_atoms)


class MoleculeDataset(Dataset):
    def __init__(self, data_frame, unique_atomic_numbers, indices=None):
        """
        Custom dataset for molecular graphs.

        Args:
            data_frame (pd.DataFrame): DataFrame with columns ['smiles', 'target'].
            unique_atomic_numbers (list[int]): List of unique atomic numbers in the dataset.
            indices (list[int], optional): List of indices to use as a subset of the dataset.
        """
        super().__init__()
        self.data_frame = data_frame
        self.unique_atomic_numbers = unique_atomic_numbers
        self._indices = indices if indices is not None else range(len(data_frame))

    def len(self):
        """
        Returns the number of examples in the dataset.
        """
        return len(self._indices)

    def get(self, idx):
        """
        Returns the data object (graph) for a given index.

        Args:
            idx (int): Index of the data to retrieve.

        Returns:
            Data: A PyTorch Geometric Data object.
        """
        actual_idx = self._indices[idx]
        smiles = self.data_frame.iloc[actual_idx]['Canonical_SMILES']
        target = self.data_frame.iloc[actual_idx]['Toxicity_Value']
        print(smiles,target)
        return molecule_to_graph(smiles, target, self.unique_atomic_numbers)
