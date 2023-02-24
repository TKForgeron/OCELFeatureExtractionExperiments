# SELF REMINDER. Copy the 'ocpa' directory to the forked one from github, so that I can push updates to github.
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

with open(f"{STORAGE_PATH}BPI2017-ocel.pkl", "rb") as file:
    ocel = pickle.load(file)

activities = list(set(ocel.log.log["event_activity"].tolist()))
feature_set = [
    TARGET_LABEL,
    (feature_factory.EVENT_PREVIOUS_TYPE_COUNT, ("GDSRCPT",)),
    (feature_factory.EVENT_ELAPSED_TIME, ()),
] + [(feature_factory.EVENT_PRECEDING_ACTIVITES, (act,)) for act in activities]

feature_storage = feature_factory.apply(
    ocel,
    scaler=PowerTransformer,
    target_label=TARGET_LABEL,
    event_based_features=feature_set,
    execution_based_features=[],
)
feature_storage.extract_normalized_train_test_split(
    test_size=0.3,
    target_label=TARGET_LABEL,
    state=RANDOM_SEED,
)

# DUMPING
with open(f"{STORAGE_PATH}BPI2017-feature_storage-split.pkl", "wb") as file:
    pickle.dump(feature_storage, file)

# LOADING
with open(f"{STORAGE_PATH}BPI2017-feature_storage-split.pkl", "rb") as file:
    feature_storage = pickle.load(file)
with open(
    f"{STORAGE_PATH}BPI2017-feature_storage-split(nolabel_when_scaling).pkl", "rb"
) as file:
    fs_nolabel_when_scaling = pickle.load(file)


def check_prop(fgs0: list, fgs1: list) -> bool:
    """Function that checks whether all node values of two lists of feature graphs are equal."""
    props = [True]
    for fg0, fg1 in zip(fgs0, fgs1):
        for n0, n1 in zip(fg0.nodes, fg1.nodes):
            prop = n0.attributes[TARGET_LABEL] == n1.attributes[TARGET_LABEL]
            props = [props[0] and prop]

    return props[0]


check_prop(fs_nolabel_when_scaling.feature_graphs, feature_storage.feature_graphs)
