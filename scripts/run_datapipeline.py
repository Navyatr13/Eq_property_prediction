from data_processing.read_data import read_smiles_data
from data_processing.mol_to_graph import molecule_to_graph
from rdkit import Chem
import torch
import os
from joblib import Parallel, delayed
import logging
import warnings
from rdkit import RDLogger
from tqdm import tqdm
from utils import check_directory, is_valid_smiles, process_smiles_to_graph
from config import DATASET_PATHS

# Suppress all warnings
warnings.filterwarnings("ignore")
# Suppress RDKit-specific warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
logging.basicConfig(filename="logs/pipeline.log", level=logging.INFO, format="%(asctime)s - %(message)s")

def run_pipeline(input_path, output_path, batch_size=100):
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


def process_dataset(input_path, out_path):
    """
    A reusable function to process a dataset only if the output directory exists and is not empty.
    """
    # Check if the output directory exists
    if os.path.exists(out_path) and os.listdir(out_path):
        print(f"Directory '{out_path}' already exists and is not empty. Skipping processing.")
    else:
        print(f"Processing dataset. Directory '{out_path}' is empty or does not exist.")
        check_directory(out_path)  # Ensure the output directory exists
        output_path = os.path.join(out_path, "graphs")  # Create the graphs directory path
        run_pipeline(input_path, output_path)  # Run the pipeline

if __name__ == "__main__":
    for dataset, paths in DATASET_PATHS.items():
        input_path = paths["input"]
        out_path = paths["output"]
        print(f"Processing {dataset} dataset...")
        process_dataset(input_path, out_path)
