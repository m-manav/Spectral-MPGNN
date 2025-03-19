import numpy as np
import torch
import lightning as L
from torch.utils.data import SubsetRandomSampler
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path

from ..data.Custom_DataSet import CustomDataset
from ..data.InputNormalize import InputNormalizer
from ..models.model_L import construct_model


def train(config):
    # Initialize Wandb for logging experiment metrics and results
    logger = WandbLogger(name="spectral-MPGNN", save_dir=config.dirs.log_dir)

    # Set up model checkpointing to save the best models based on validation loss
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.dirs.checkpoint_dir,
        monitor="val_loss",
        filename="1Dbar-{epoch:02d}-{val_loss:.2f}",
        save_top_k=10,
        mode="min",
    )

    # Enable early stopping if validation loss doesn't improve for 5 epochs
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, mode="min", verbose=False
    )

    # ---------------------------
    # Prepare Training Data
    # ---------------------------
    train_config = config.train
    train_data_size_to_use = train_config.train_data_size_to_use
    processed_input_data_dir = config.data.processed_input_data_dir

    # Load the preprocessed training data
    processed_dataset = torch.load(
        Path(processed_input_data_dir) / Path("processed_train_set.pt"),
        weights_only=False,
    )
    data_size = len(processed_dataset["spt_data_list"])

    # Split the data into training and validation sets (90% train, 10% validation)
    val_data_size = int(0.1 * data_size)
    train_data_size = int(min(train_data_size_to_use, 0.9 * data_size))

    # Extract spatial (spa) and spectral (spc) training data
    spa_train_dataset = processed_dataset["spt_data_list"][:train_data_size]
    spc_train_dataset = processed_dataset["spc_data_list"][:train_data_size]

    # Normalize the spatial input data for better training stability
    spa_input_normalizer = InputNormalizer(spa_train_dataset)
    spa_train_dataset_normal = spa_input_normalizer.normalize_data(spa_train_dataset)

    # Create training dataset and use a sampler for random batch selection
    train_dataset = CustomDataset(spa_train_dataset_normal, spc_train_dataset)
    indices = np.arange(train_data_size)
    sampler = SubsetRandomSampler(indices)
    train_loader = DataLoader(
        train_dataset,
        num_workers=train_config.num_workers,
        batch_size=train_config.batch_size,
        sampler=sampler,
    )

    # Extract and normalize validation data
    spa_val_dataset = processed_dataset["spt_data_list"][
        train_data_size : train_data_size + val_data_size
    ]
    spc_val_dataset = processed_dataset["spc_data_list"][
        train_data_size : train_data_size + val_data_size
    ]
    spa_val_dataset_normal = spa_input_normalizer.normalize_data(spa_val_dataset)
    val_dataset = CustomDataset(spa_val_dataset_normal, spc_val_dataset)
    val_loader = DataLoader(
        val_dataset,
        num_workers=train_config.num_workers,
        batch_size=train_config.batch_size,
        shuffle=False,
    )

    # ---------------------------
    # Model Definition
    # ---------------------------

    # Extract sample data to infer input/output dimensions for the model
    spa_data_ex1 = processed_dataset["spt_data_list"][0]
    spc_data_ex1 = processed_dataset["spc_data_list"][0]
    config.model.input_dim_node = spa_data_ex1.x.shape[1]
    config.model.input_dim_edge = spa_data_ex1.edge_attr.shape[1]
    config.model.spc_input_dim_node = spc_data_ex1.x.shape[1]
    config.model.spc_input_dim_edge = spc_data_ex1.edge_attr.shape[1]
    config.model.output_dim = spa_data_ex1.y.shape[1]

    # Build the model using the configuration
    model = construct_model(config)

    # ---------------------------
    # Training Setup
    # ---------------------------

    # Configure the PyTorch Lightning trainer for model training and evaluation
    trainer = L.Trainer(
        max_epochs=train_config.num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        # fast_dev_run=True,
    )

    # Train the model using the training and validation data
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ---------------------------
    # Test Data Preparation and Evaluation
    # ---------------------------

    # Load and normalize the preprocessed test data
    processed_test_dataset = torch.load(
        Path(processed_input_data_dir) / Path("processed_test_set.pt"),
        weights_only=False,
    )
    spa_test_dataset = processed_test_dataset["spt_data_list"]
    spc_test_dataset = processed_test_dataset["spc_data_list"]
    spa_test_dataset_normal = spa_input_normalizer.normalize_data(spa_test_dataset)

    # Create the test dataset and loader
    test_dataset = CustomDataset(spa_test_dataset_normal, spc_test_dataset)
    test_loader = DataLoader(
        test_dataset, batch_size=train_config.batch_size, shuffle=True
    )

    # Evaluate the best checkpointed model on the test data
    trainer.test(dataloaders=test_loader, ckpt_path="best")
