# %%
# SELF REMINDER. Copy the 'ocpa' directory to the forked one from github, so that I can push updates to github.

# Python native
import pickle
from statistics import median as median
from tqdm import tqdm
import os

os.chdir("c:\\Users\\Tim\\Development\\OCELFeatureExtractionExperiments")
from copy import deepcopy

# Data handling
import pandas as pd
import numpy as np

# Object centric process mining
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage

# PyG
import torch
from ocpa_PyG_integration.EventGraphDataset import EventGraphDataset


# %%
def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %%
ds_train = EventGraphDataset(
    root="data/ocpa-processed/",
    filename="BPI2017-feature_storage_split.pkl",
    label_key=("event_remaining_time", ()),
    train=True,
)

# %%
ds_test = EventGraphDataset(
    root="data/ocpa-processed/",
    filename="BPI2017-feature_storage_split.pkl",
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
model = model.to(device)
# data = ds_train.to(device)

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
NUM_GRAPHS_PER_BATCH = 512
# Define loss function (CrossEntropyLoss for Classification Problems with
# probability distributions)
criterion = torch.nn.MSELoss()

train_loader = DataLoader(ds_train, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(ds_test, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n MAE: \n {mean_absolute_error(y_pred, y_true)}")
    print(f"MSE: {mean_squared_error(y_true, y_pred)}")
    print(f"R^2: {r2_score(y_true, y_pred)}")


# %%


def train(epoch):
    all_preds = []
    all_labels = []

    model.train()
    optimizer.zero_grad()
    # Use all data as input, because all nodes have node features
    out = model(data.x, data.edge_index)
    # Only use nodes with labels available for loss calculation --> mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(batch.x.float(), batch.edge_index, batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()
        # Update tracking
        running_loss += loss.item()
        step += 1
        print(f"{running_loss} \r", end=" ")
        print(f"{step} \r", end=" ")
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    calculate_metrics(all_preds, all_labels, epoch, "train")
    return running_loss / step


def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_preds_raw = []
    all_labels = []
    running_loss = 0.0
    step = 0
    for batch in test_loader:
        batch.to(device)
        pred = model(batch.x.float(), batch.edge_index, batch.batch)
        loss = loss_fn(torch.squeeze(pred), batch.y.float())

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(np.rint(torch.sigmoid(pred).cpu().detach().numpy()))
        all_preds_raw.append(torch.sigmoid(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    print(all_preds_raw[0][:10])
    print(all_preds[:10])
    print(all_labels[:10])
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return running_loss / step


def run_training(loss_fn):
    # Start training
    best_loss = 1000
    early_stopping_counter = 0
    for epoch in range(300):
        if early_stopping_counter <= 10:  # = x * 5
            # Training
            model.train()
            loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
            print(f"Epoch {epoch} | Train Loss {loss}")

            # Testing
            model.eval()
            if epoch % 5 == 0:
                loss = test(epoch, model, test_loader, loss_fn)
                print(f"Epoch {epoch} | Test Loss {loss}")

                # Update best loss
                if float(loss) < best_loss:
                    best_loss = loss
                    # # Save the currently best model
                    # mlflow.pytorch.log_model(model, "model", signature=SIGNATURE)
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1

        else:
            print("Early stopping due to no improvement.")
            return [best_loss]


# %%
results = run_training(criterion)
