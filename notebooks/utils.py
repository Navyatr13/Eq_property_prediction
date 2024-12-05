from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

def convert_to_canonical_smiles(smiles):
    try:
        # Convert input SMILES to an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        # Convert the molecule to a canonical SMILES string
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        return canonical_smiles
    except Exception as e:
        return f"Error: {e}"

def normalize_tautomers(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Normalize the molecule (tautomer handling included)
        normalizer = rdMolStandardize.Normalize()
        mol = normalizer.normalize(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        return None

def normalize_tautomers(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Normalize the molecule (tautomer handling included)
        normalizer = rdMolStandardize.Normalize()
        mol = normalizer.normalize(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        return None

def remove_salts(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Remove small fragments or salts
        fragment_remover = rdMolStandardize.FragmentRemover()
        mol = fragment_remover.remove(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        return None


def neutralize_charges(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Neutralize charges
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        return None

def standardize_functional_groups(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # Apply template-based transformations if needed
        # Example: Aromatization
        AllChem.Kekulize(mol, clearAromaticFlags=True)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        return None

def add_hydrogens(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        return None

def standardize_stereochemistry(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        return None

def standardize_smiles_pipeline(smiles):
    print(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    uncharger = rdMolStandardize.Uncharger()
    fragment_remover = rdMolStandardize.FragmentRemover()

    # Normalize, remove salts, uncharge, and canonicalize
    mol = normalize_smiles(smiles)
    mol = fragment_remover.remove(mol)
    mol = uncharger.uncharge(mol)
    return Chem.MolToSmiles(mol, canonical=True)

def validate_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return smiles  # True if valid, False otherwise
    except Exception as e:
        return False

# Define the function to normalize a SMILES string
def normalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid SMILES string"
        # Create a normalizer object
        normalizer = rdMolStandardize.Normalizer()
        # Normalize the molecule
        normalized_mol = normalizer.normalize(mol)
        return normalized_mol #Chem.MolToSmiles(normalized_mol, canonical=True)
    except Exception as e:
        return f"Error: {e}"