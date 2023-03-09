# %%
# SELF REMINDER. Copy the 'ocpa' directory to the forked one from github, so that I can push updates to github.

# Python native
import pickle
from statistics import median as median
from tqdm import tqdm
import os

os.chdir("/home/tim/Development/OCELFeatureExtractionExperiments/")
from copy import deepcopy

# Data handling
import pandas as pd
import numpy as np

# Object centric process mining
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage

# PyG
import torch
from ocpa_PyG_integration.EventGraphDataset import EventGraphDataset

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Global variables
from experiment_config import STORAGE_PATH, RANDOM_SEED, TARGET_LABEL


# %%
def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
ds_train = EventGraphDataset(
    root="data/ocpa-processed/",
    filename="BPI2017-feature_storage-split.pkl",
    label_key=("event_remaining_time", ()),
    train=True,
)

# %%
ds_test = EventGraphDataset(
    root="data/ocpa-processed/",
    filename="BPI2017-feature_storage-split.pkl",
    label_key=("event_remaining_time", ()),
    test=True,
)

# %%
ds_train.get_summary()

# %%
from model import GCN
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm


# Initialize model
model = GCN(ds_train.num_node_features, 12)
print(model)
print(f"Number of parameters: {count_parameters(model)}")


# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = model.to(device)
# data = ds_train.to(device)

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
NUM_GRAPHS_PER_BATCH = 512
# Define loss function (CrossEntropyLoss for Classification Problems with
# probability distributions)
loss_fn = torch.nn.MSELoss()

train_loader = DataLoader(ds_train, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n MAE: \n {mean_absolute_error(y_pred, y_true)}")
    print(f"MSE: {mean_squared_error(y_true, y_pred)}")
    print(f"R^2: {r2_score(y_true, y_pred)}")


# %%
def train_one_epoch(
    epoch_index: int, model, train_loader, optimizer, loss_fn, tb_writer
):
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
        # Reset gradients
        optimizer.zero_grad()
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
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


# %%
# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"runs/ocel_trainer_{timestamp}")

EPOCHS = 50

best_vloss = 1_000_000.0

for epoch in range(EPOCHS):
    print("EPOCH {}:".format(epoch + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn, writer)

    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    for i, vdata in enumerate(test_loader):
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
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

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
        model_path = "model_{}_{}".format(timestamp, epoch)
        torch.save(model.state_dict(), model_path)

# %%
