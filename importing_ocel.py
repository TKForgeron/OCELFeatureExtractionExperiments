# SELF REMINDER. Copy the 'ocpa' directory to the forked one from github, so that I can push updates to github.
# Timing
import timeit

start = timeit.default_timer()


# Python native
import pickle
from statistics import median as median

# Data handling
# Object centric process mining
# import ocpa.algo.evaluation.precision_and_fitness.utils as evaluation_utils # COMMENTED OUT BY TIM
# import ocpa.algo.evaluation.precision_and_fitness.evaluator as precision_fitness_evaluator # COMMENTED OUT BY TIM
import ocpa.objects.log.importer.csv.factory as csv_import_factory
import ocpa.algo.predictive_monitoring.factory as feature_factory

# # Simple machine learning models, procedure tools, and evaluation metrics
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer

# # Custom GNN tools
# from gnn_utils import (
#     generate_graph_dataset,
#     # get_ordered_event_list,
#     # visualize_graph,
#     # show_remaining_times,
#     # visualize_instance,
#     # GraphDataLoader,
#     # GCN,
#     # evaluate_gnn,
# )

# Global variables
from experiment_config import STORAGE_PATH, RANDOM_SEED, TARGET_LABEL

filename = "example_logs/mdl/BPI2017-Final.csv"
object_types = ["application", "offer"]
parameters = {
    "obj_names": object_types,
    "val_names": [],
    "act_name": "event_activity",
    "time_name": "event_timestamp",
    "sep": ",",
}
file_path_object_attribute_table = None

ocel = csv_import_factory.apply(
    filename, csv_import_factory.TO_OCEL, parameters, file_path_object_attribute_table
)

activities = list(set(ocel.log.log["event_activity"].tolist()))
feature_set = [
    TARGET_LABEL,
    (feature_factory.EVENT_PREVIOUS_TYPE_COUNT, ("GDSRCPT",)),
    (feature_factory.EVENT_ELAPSED_TIME, ()),
] + [(feature_factory.EVENT_PRECEDING_ACTIVITES, (act,)) for act in activities]
feature_storage = feature_factory.apply(
    ocel,
    event_based_features=feature_set,
    execution_based_features=[],
)
feature_storage.extract_normalized_train_test_split(
    test_size=0.3,
    validation_size=0.2,
    scaler=StandardScaler,
    scaling_exempt_features=[TARGET_LABEL],
    state=RANDOM_SEED,
)

with open(f"{STORAGE_PATH}BPI2017-feature_storage-split.pkl", "wb") as file:
    pickle.dump(feature_storage, file)

# # keep list of first three events for comparability of regression use case
# events_to_remove = []
# for g in tqdm(feature_storage.feature_graphs):
#     event_ids = [n.event_id for n in g.nodes]
#     event_ids.sort()
#     events_to_remove = events_to_remove + event_ids[:3]

# label_order = None
# accuracy_dict = {}

# train_idx, val_idx = train_test_split(feature_storage.training_indices, test_size=0.2)
# x_train, y_train = generate_graph_dataset(
#     feature_storage.feature_graphs, train_idx, ocel
# )
# x_val, y_val = generate_graph_dataset(feature_storage.feature_graphs, val_idx, ocel)
# x_test, y_test = generate_graph_dataset(
#     feature_storage.feature_graphs, feature_storage.test_indices, ocel
# )
elapsed_time = timeit.default_timer() - start
minutes, seconds = int(elapsed_time / 60), int(elapsed_time % 60)
print(f"Time taken: {minutes}m{seconds}s")
