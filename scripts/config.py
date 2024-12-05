# config.py

unique_atomic_numbers = [1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 44, 46, 48, 50, 53, 56, 60, 78, 79, 80, 81, 83]
selected_model = "GNNModel"

DATASET_PATHS = {
    "train": {
        "input": "./data/New_train.csv",
        "output": "./data/processed_train"
    },
    "val": {
        "input": "./data/New_val.csv",
        "output": "./data/processed_val"
    },
    "test": {
        "input": "./data/New_test.csv",
        "output": "./data/processed_test"
    }
}

MODELS = {
    "GNNModel": {
        "input_dim": 49,
        "hidden_dim": 512,
        "output_dim": 1,
        "learning_rate": 8.52E-05,
        "dropout_rate": 0.15,
        "weight_decay": 0.002299627
    },
    "MPNNModel": {
        "input_dim": 6,
        "edge_dim": 3,
        "hidden_dim": 128,
        "output_dim": 2,
        "learning_rate": 0.001,
        "dropout_rate": 0.36
    }
}


TRAINING_CONFIG = {
    "max_epochs": 10,
    "gradient_clip_val": 1.0,
    "train_batch_size": 256,
    "val_batch_size": 512,
    "test_batch_size": 512,
    "early_stopping_patience": 10,
    "monitor_metric": "val_loss",
    "monitor_mode": "min"
}