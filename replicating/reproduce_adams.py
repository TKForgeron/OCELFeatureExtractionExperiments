# SELF REMINDER. Copy the 'ocpa' directory to the forked one from github, so that I can push updates to github.

# Python native
import pickle
from statistics import median as median
from tqdm import tqdm
import os

os.chdir("/home/tim/Development/OCELFeatureExtractionExperiments/")
# from copy import deepcopy

# Object centric process mining
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage
import ocpa.algo.predictive_monitoring.factory as feature_factory

# Data handling
# import pandas as pd
# import numpy as np

# # Simple machine learning models, procedure tools, and evaluation metrics
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# PyG
import torch

# from replicating.ocpa_PyG_integration.EventGraphDataset import EventGraphDataset
from ocpa_PyG_integration.EventSubGraphDataset import EventSubGraphDataset

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Global variables
# from replicating.experiment_config import STORAGE_PATH, RANDOM_SEED, TARGET_LABEL
STORAGE_PATH = "data/ocpa-processed"
FEATURE_STORAGE_FILE = "BPI17-scaled-split.fs"
RANDOM_SEED = 42
TARGET_LABEL = (feature_factory.EVENT_REMAINING_TIME, ())

# If FEATURE_STORAGE_FILE not cached, generate it
if not os.path.exists(f"{STORAGE_PATH}/raw/{FEATURE_STORAGE_FILE}"):
    with open(
        f"{STORAGE_PATH}/raw/BPI17-feature_storage-[C2,D1,P2,P3,O3].fs", "rb"
    ) as file:
        feature_storage: FeatureStorage = pickle.load(file)

    feature_storage.extract_normalized_train_test_split(
        test_size=0.3,
        validation_size=0.2,
        scaler=StandardScaler,
        scaling_exempt_features=[],
        state=RANDOM_SEED,
    )

    with open(
        f"{STORAGE_PATH}/raw/{FEATURE_STORAGE_FILE}",
        "wb",
    ) as file:
        pickle.dump(feature_storage, file)

ds_train = EventSubGraphDataset(
    root=STORAGE_PATH,
    filename=FEATURE_STORAGE_FILE,
    label_key=TARGET_LABEL,
    size_subgraph_samples=4,
    train=True,
    verbosity=51,
)
ds_val = EventSubGraphDataset(
    root=STORAGE_PATH,
    filename=FEATURE_STORAGE_FILE,
    label_key=TARGET_LABEL,
    size_subgraph_samples=4,
    validation=True,
    verbosity=51,
)
ds_test = EventSubGraphDataset(
    root=STORAGE_PATH,
    filename=FEATURE_STORAGE_FILE,
    label_key=TARGET_LABEL,
    size_subgraph_samples=4,
    test=True,
    verbosity=51,
)


# print("Train set")
# print(ds_train.get_summary())
# print()

# # # Validation set gives error for now
# print("Validation set")
# print(ds_val.get_summary())
# print()

# print("Test set")
# print(ds_test.get_summary())
# print()


from model import GCN, GAT
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Initialize model
model = GCN(
    ds_train.num_node_features, {"num_hidden_features": ds_train.num_node_features}
)
print(model)
print(f"Number of parameters: {count_parameters(model)}")


# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = model.to(device)
# data = ds_train.to(device)

# Initialize Optimizer
learning_rate = 0.01
# decay = 5e-4
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
    #  weight_decay=decay
)
NUM_GRAPHS_PER_BATCH = 64

mse = torch.nn.MSELoss()
mae = torch.nn.L1Loss()
loss_fn = mae

train_loader = DataLoader(ds_train, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
val_loader = DataLoader(ds_val, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=128, shuffle=True)


def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n MAE: \n {mean_absolute_error(y_pred, y_true)}")
    print(f"MSE: {mean_squared_error(y_true, y_pred)}")
    print(f"R^2: {r2_score(y_true, y_pred)}")


def train_one_epoch(
    epoch_index: int, model, train_loader, optimizer, loss_fn, tb_writer
) -> float:
    # Enumerate over the data
    running_loss = 0.0
    last_loss = 0
    for i, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        # Every data instance is an input + label pair
        inputs, adjacency_matrix, labels = (
            batch.x.float(),
            batch.edge_index,
            batch.y.float(),
        )
        # Reset gradients (set_to_none is faster than to zero)
        optimizer.zero_grad(set_to_none=True)
        # Passing the node features and the connection info
        outputs = model(inputs, adjacency_matrix)
        # Compute loss and gradients
        loss = loss_fn(torch.squeeze(outputs), labels)
        loss.backward()
        # Adjust learnable weights
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print(f"  batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def run_training(
    num_epochs, model, train_loader, validation_loader, optimizer, loss_fn, timestamp
):
    model_path = f"models/{model.get_class_name()}_{timestamp}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    writer = SummaryWriter(f"{model_path}/run")
    best_vloss = 1_000_000_000_000_000.0

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1}:")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, loss_fn, writer
        )

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            vdata.to(device)
            vinputs, vadjacency_matrix, vlabels = (
                vdata.x.float(),
                vdata.edge_index,
                vdata.y.float(),
            )
            voutputs = model(vinputs, vadjacency_matrix)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print(f"LOSS train {avg_loss} valid {avg_vloss}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), f"{model_path}/state_dict_epoch{epoch}.pt")


# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%Hh%Mm")
EPOCHS = 30

run_training(
    num_epochs=EPOCHS,
    model=model,
    train_loader=train_loader,
    validation_loader=val_loader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    timestamp=timestamp,
)
