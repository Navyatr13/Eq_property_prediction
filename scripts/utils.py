import random
import numpy as np
import torch
import os
from rdkit import Chem
import logging
from data_processing import molecule_to_graph
from rdkit.Chem import AllChem
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False
def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_smiles_to_graph(row):
    """
    Converts a SMILES string into a graph.
    Returns a tuple: (graph, success_flag)
    """
    try:
        graph = molecule_to_graph(row.Canonical_SMILES, row.Toxicity_Value)
        return graph, True  # Success
    except Exception as e:
        logging.error(f"Error processing SMILES {row.Canonical_SMILES}: {e}")
        return None, False  # Failure

def check_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")
    else:
        print("Check if you want to overwrite the folder")
        print(f"Directory already exists: {path}")
        return False