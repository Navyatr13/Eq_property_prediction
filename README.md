# Equivariance Property Prediction

This project is focused on processing molecular SMILES data, converting it to graph structures, and training Graph Neural Network (GNN) models for property prediction. It includes modules for data processing, training models, hyperparameter optimization, and visualizations.

---

## Dependencies

Ensure you have the following Python libraries installed:

- `torch`: For building and training neural networks
- `torch-geometric`: For graph neural network implementations
- `pytorch-lightning`: For streamlined PyTorch training loops
- `matplotlib`: For visualizing training and evaluation metrics
- `rdkit`: For processing SMILES strings
- `joblib`: For parallel processing
- `tqdm`: For progress bars
- `scikit-learn`: For t-SNE and UMAP visualizations

Install them via pip:
```
pip install torch torch-geometric pytorch-lightning matplotlib rdkit joblib tqdm scikit-learn
```
### Usage
1. Run the Data Pipeline
Use the run_datapipeline.py script to process SMILES data into graph structures and save the results in batches.
``` 
python scripts/run_datapipeline.py 
```
This script:

Reads raw SMILES data (New_train.csv, New_val.csv, and New_test.csv).
Converts them into graph representations.
Saves the graphs in .pt files for each batch.
Modify the input_path and output_path variables in the script to point to your datasets.

2. Train the GNN Model
Train the GNN model using the processed graph data with the train_gnn.py script.

```
python scripts/train_gnn.py
```
This script:

Loads the processed train and validation graph data.
Defines a GNN model (GNNModel by default) with specified hyperparameters.
Trains the model using PyTorch Lightning.
Includes early stopping based on validation loss.

Hyperparameters:

Input features: input_dim
Hidden layer size: hidden_dim
Learning rate: learning_rate
Dropout: dropout_rate
Weight decay: weight_decay

3. Optimize Hyperparameters
Run hyper_parameter.py to perform hyperparameter optimization via grid search.

``` python scripts/hyper_parameter.py ```
This script:

Performs a grid search to find the best model hyperparameters.
Uses validation loss to evaluate hyperparameter combinations.

4. Visualize Results
Use visualization.py to generate plots for losses, ROC curves, and more.

``` python scripts/visualization.py ```

This script:

Visualizes training losses.
Plots ROC curves to assess classification performance.

### Key Modules
##### 1. Data Processing
Located in data_processing/, these modules handle SMILES data loading and graph conversion:

read_data.py: Reads SMILES strings and their labels into a pandas DataFrame.
mol_to_graph.py: Converts molecular data into graph structures.
##### 2. Models
Located in models/, this directory contains various GNN implementations:

gnn.py: Implements a standard GNN model.
mpnn.py: Implements a Message Passing Neural Network (MPNN).
##### 3. Training
The train_gnn.py script in scripts/ handles model training:

Utilizes pytorch-lightning for efficient model training.
Includes early stopping for improved training efficiency.
##### 4. Visualization
The visualization.py script provides:

Loss plots over training epochs.
ROC curves for model evaluation.

### Example Workflow
Process SMILES Data:
``` python scripts/run_datapipeline.py ```

Train the GNN Model:
``` python scripts/train_gnn.py ```
Visualize Results:

``` python scripts/visualization.py ```
Optimize Hyperparameters:

``` python scripts/hyper_parameter.py```
Authors
This project was created by Navya Ramesh. For inquiries, please contact: navyatr06@gmail.com .


