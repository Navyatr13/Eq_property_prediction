from rdkit import Chem
from rdkit.Chem import AllChem
#from rdkit.Chem import rdMolStandardize

def clean_smiles(smiles):
    """
    Cleans and standardizes a SMILES string.
    """
    try:
        # Validate SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add explicit hydrogens
        mol = Chem.AddHs(mol)

        # Canonicalize
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # Standardize aromaticity
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)

        # Standardize stereochemistry
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Normalize tautomers
        #enumerator = rdMolStandardize.TautomerEnumerator()
        #mol = enumerator.Canonicalize(mol)

        # Remove salts and fragments
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)

        # Neutralize charges
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Return final standardized SMILES
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    except Exception as e:
        print(f"Failed to clean SMILES {smiles}: {e}")
        return None
