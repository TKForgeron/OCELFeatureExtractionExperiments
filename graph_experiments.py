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

filename = "example_logs/mdl/BPI2017-Final.csv"
object_types = ["application", "offer"]
parameters = {
    "obj_names": object_types,
    "val_names": [],
    "act_name": "event_activity",
    "time_name": "event_timestamp",
    "sep": ",",
    "take_sample": 9999,
}
file_path_object_attribute_table = None

ocel = csv_import_factory.apply(
    filename, csv_import_factory.TO_OCEL, parameters, file_path_object_attribute_table
)

activities = list(set(ocel.log.log["event_activity"].tolist()))
feature_set = [
    (feature_extractor.EVENT_REMAINING_TIME, ()),
    (feature_extractor.EVENT_PREVIOUS_TYPE_COUNT, ("GDSRCPT",)),
    (feature_extractor.EVENT_ELAPSED_TIME, ()),
] + [(feature_extractor.EVENT_PRECEDING_ACTIVITES, (act,)) for act in activities]
feature_storage = feature_extractor.apply(ocel, feature_set, [])

# # Pickle feature storage objects
# with open(f"data/processed/BPI2017-feature_storage.pkl", "wb") as file:
#     pickle.dump(feature_storage, file)

# with open(f"data/processed/BPI2017-ocel.pkl", "wb") as file:
#     pickle.dump(ocel, file)


# # Load pickled feature storage objects
# with open(f"data/processed/BPI2017-feature_storage.pkl", "rb") as file:
#     feature_storage = pickle.load(file)

# with open(f"data/processed/BPI2017-ocel.pkl", "rb") as file:
#     ocel = pickle.load(file)

feature_storage.extract_normalized_train_test_split(0.3, state=42)

# keep list of first three events for comparability of regression use case
events_to_remove = []
for g in tqdm(feature_storage.feature_graphs):
    event_ids = [n.event_id for n in g.nodes]
    event_ids.sort()
    events_to_remove = events_to_remove + event_ids[:3]

label_order = None
accuracy_dict = {}

train_idx, val_idx = train_test_split(feature_storage.training_indices, test_size=0.2)
x_train, y_train = generate_graph_dataset(
    feature_storage.feature_graphs, train_idx, ocel
)
x_val, y_val = generate_graph_dataset(feature_storage.feature_graphs, val_idx, ocel)
x_test, y_test = generate_graph_dataset(
    feature_storage.feature_graphs, feature_storage.test_indices, ocel
)
