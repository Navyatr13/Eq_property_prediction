from data_processing.read_data import read_smiles_data
from data_processing.mol_to_graph import molecule_to_graph
from rdkit import Chem
import torch
from joblib import Parallel, delayed
import logging
import warnings
from rdkit import RDLogger
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")
# Suppress RDKit-specific warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
logging.basicConfig(filename="logs/pipeline.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except Exception:
        return False

def process_smiles_to_graph(row):
    """
    Converts a SMILES string into a graph.
    Returns a tuple: (graph, success_flag)
    """
    try:
        graph = molecule_to_graph(row.Canonical_SMILES, row.Toxicity_Value)
        return graph, True  # Success
    except Exception as e:
        logging.error(f"Error processing SMILES {row.SMILES}: {e}")
    #    return None, False  # Failure

def run_pipeline(input_path, output_path, batch_size=1000):
    data = read_smiles_data(input_path)
    logging.info(f"Loaded dataset with {len(data)} rows.")

    # Filter invalid SMILES
    data = data[data['Canonical_SMILES'].apply(is_valid_smiles)]
    logging.info(f"Filtered dataset size: {len(data)}")

    total_success = 0
    total_failure = 0

    for start in tqdm(range(0, len(data), batch_size)):
        batch = data.iloc[start:start + batch_size]
        logging.info(f"Processing batch {start // batch_size + 1} of {len(data) // batch_size + 1}...")

        # Parallel batch processing
        results = Parallel(n_jobs=-1)(
            delayed(process_smiles_to_graph)(row) for row in batch.itertuples(index=False)
        )

        # Separate successful and failed results
        batch_graphs = [res[0] for res in results if res[1] and res[0] is not None]  # Ensure non-None
        batch_failures = [res for res in results if not res[1]]  # Collect failures for logging

        # Update counts
        total_success += len(batch_graphs)
        total_failure += len(batch_failures)

        # Log failure count
        logging.info(f"Batch {start // batch_size + 1}: {len(batch_failures)} failed graphs.")

        # Save intermediate results
        if batch_graphs:  # Save only if there are valid graphs
            batch_output_path = f"{output_path}_batch_{start // batch_size + 1}.pt"
            torch.save(batch_graphs, batch_output_path)
            logging.info(f"Saved batch {start // batch_size + 1} to {batch_output_path}")
            logging.info(f"Saved {len(batch_graphs)} to batch {start // batch_size + 1}")

    # Final log of totals
    logging.info(f"Total successful graphs: {total_success}")
    logging.info(f"Total failed graphs: {total_failure}")
    print(f"Total successful graphs: {total_success}")
    print(f"Total failed graphs: {total_failure}")


if __name__ == "__main__":
    
    input_path = "D:/Equivarance_property_prediction/data/New_train.csv"
    output_path = "D:/Equivarance_property_prediction/data/processed_train/graphs"
    run_pipeline(input_path, output_path)
    
    input_path = "D:/Equivarance_property_prediction/data/New_val.csv"
    output_path = "D:/Equivarance_property_prediction/data/processed_val/graphs"
    run_pipeline(input_path, output_path)

    input_path = "D:/Equivarance_property_prediction/data/New_test.csv"
    output_path = "D:/Equivarance_property_prediction/data/processed_test/graphs"
    run_pipeline(input_path, output_path)
