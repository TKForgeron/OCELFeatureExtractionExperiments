# SELF REMINDER. Copy the 'ocpa' directory to the forked one from github, so that I can push updates to github.

# Python native
import time
import random
import pickle
from datetime import timedelta
from statistics import median as median
from tqdm import tqdm
from ast import literal_eval

# Data handling
import pandas as pd
import numpy as np

# Object centric process mining
from ocpa.objects.log.ocel import OCEL
import ocpa.objects.log.importer.ocel.factory as ocel_import_factory  # json/xml import factory
import ocpa.objects.log.importer.csv.factory as csv_import_factory
import ocpa.objects.log.converter.factory as convert_factory
import ocpa.algo.util.filtering.log.time_filtering
import ocpa.algo.util.filtering.log.variant_filtering as trace_filtering

# import ocpa.algo.evaluation.precision_and_fitness.utils as evaluation_utils # COMMENTED OUT BY TIM
# import ocpa.algo.evaluation.precision_and_fitness.evaluator as precision_fitness_evaluator # COMMENTED OUT BY TIM
import ocpa.algo.predictive_monitoring.factory as feature_extractor
from ocpa.algo.predictive_monitoring import time_series
from ocpa.algo.predictive_monitoring import tabular, sequential
from ocpa.algo.discovery.ocpn import algorithm as ocpn_discovery_factory
import ocpa.visualization.oc_petri_net.factory as vis_factory
import ocpa.visualization.log.variants.factory as log_viz

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Simple machine learning models and procedure tools and evaluation metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score
import shap

# Tensorflow deep learning
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import keras.backend as K

# Custom GNN tools
from gnn_utils import (
    generate_graph_dataset,
    get_ordered_event_list,
    visualize_graph,
    show_remaining_times,
    visualize_instance,
    GraphDataLoader,
    GCN,
    evaluate_gnn,
)


# Load pickled feature storage objects
with open(f"data/processed/BPI2017-train_loader.pkl", "rb") as file:
    train_loader = pickle.load(file)
with open(f"data/processed/BPI2017-val_loader.pkl", "rb") as file:
    val_loader = pickle.load(file)
with open(f"data/processed/BPI2017-test_loader.pkl", "rb") as file:
    test_loader = pickle.load(file)

# Defining GCN model
tf.keras.backend.clear_session()
model = GCN(24, 24)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss_function = tf.keras.losses.MeanAbsoluteError()

# run tensorflow training loop
epochs = 30
iter_idx = np.arange(0, train_loader.__len__())
loss_history = []
val_loss_history = []
step_losses = []
for epoch in range(epochs):
    print("Running epoch:", epoch)
    np.random.shuffle(iter_idx)
    current_loss = step = 0
    for batch_id in tqdm(iter_idx):
        step += 1
        dgl_batch, label_batch = train_loader.__getitem__(batch_id)
        with tf.GradientTape() as tape:
            pred = model(dgl_batch, dgl_batch.ndata["features"])
            loss = loss_function(label_batch, pred)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        step_losses.append(loss.numpy())
        current_loss += loss.numpy()
        # if (step % 100 == 0): print('Loss: %s'%((current_loss / step)))
        loss_history.append(current_loss / step)
    val_predictions, val_labels = evaluate_gnn(val_loader, model)
    val_loss = tf.keras.metrics.mean_absolute_error(
        np.squeeze(val_labels), np.squeeze(val_predictions)
    ).numpy()
    print("    Validation MAE GNN:", val_loss)
    if len(val_loss_history) < 1:
        model.save_weights("gnn_checkpoint.tf")
        print("    GNN checkpoint saved.")
    else:
        if val_loss < np.min(val_loss_history):
            model.save_weights("gnn_checkpoint.tf")
            print("    GNN checkpoint saved.")
    val_loss_history.append(val_loss)

# visualize training progress
pd.DataFrame({"loss": loss_history, "step_losses": step_losses}).plot(
    subplots=True, layout=(1, 2), sharey=True
)

# restore weights from best epoch
cp_status = model.load_weights("gnn_checkpoint.tf")
cp_status.assert_consumed()
