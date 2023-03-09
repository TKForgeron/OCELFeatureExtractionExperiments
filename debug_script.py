# SELF REMINDER. Copy the 'ocpa' directory to the forked one from github, so that I can push updates to github.
# Python native
import pickle
from statistics import median, mean
import statistics
import timeit
from dataclasses import dataclass
from functools import partial

# Data handling
# Object centric process mining
# import ocpa.algo.evaluation.precision_and_fitness.utils as evaluation_utils # COMMENTED OUT BY TIM
# import ocpa.algo.evaluation.precision_and_fitness.evaluator as precision_fitness_evaluator # COMMENTED OUT BY TIM
import ocpa.objects.log.importer.csv.factory as csv_import_factory
import ocpa.algo.predictive_monitoring.factory as feature_factory
from ocpa.objects.log.ocel import OCEL, OCELslots

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
from replicating.experiment_config import STORAGE_PATH, RANDOM_SEED, TARGET_LABEL

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

# Importing OCEL
ocel = csv_import_factory.apply(
    filename, csv_import_factory.TO_OCEL, parameters, file_path_object_attribute_table
)

original_csv_columns = (
    "event_None",
    "event_Unnamed: 0",
    "event_id",
    "application",
    "event_activity",
    "event_start_timestamp",
    "event_timestamp",
    "event_LoanGoal",
    "event_ApplicationType",
    "event_RequestedAmount",
    "event_Action",
    "event_FirstWithdrawalAmount",
    "event_Accepted",
    "event_NumberOfTerms",
    "offer",
    "event_org:resource",
    "event_MonthlyCost",
    "event_EventOrigin",
    "event_EventID",
    "event_Selected",
    "event_CreditScore",
    "event_OfferedAmount",
    "event_CaseID",
)

event_attributes = [
    "event_RequestedAmount",
    "event_FirstWithdrawalAmount",
    "event_Accepted",
    "event_MonthlyCost",
    "event_Selected",
    "event_CreditScore",
    "event_OfferedAmount",
]  # Manually selected for BPI17
activities = ocel.log.log["event_activity"].unique().tolist()


feature_set = (
    [
        # (feature_factory.EVENT_DURATION),
        (feature_factory.EVENT_IDENTITY, ()),
        # (feature_factory.EVENT_TYPE_COUNT, ()),
        #
    ]
    + [(feature_factory.EVENT_CURRENT_ACTIVITIES, (act,)) for act in activities]  # C1
    + [(feature_factory.EVENT_PRECEDING_ACTIVITIES, (act,)) for act in activities]  # C2
    + [
        (feature_factory.EVENT_PREVIOUS_ACTIVITY_COUNT, (act,)) for act in activities
    ]  # C3
    + [(feature_factory.EVENT_ACTIVITY, (act,)) for act in activities]  # C5
    + [
        # (feature_factory.EVENT_AGG_PREVIOUS_CHAR_VALUES, ()),  # D1
        # (feature_factory.EVENT_PRECEDING_CHAR_VALUES, ()),  # D2
        # (feature_factory.EVENT_CHAR_VALUE, ()),  # D3
        # (feature_factory.EVENT_CURRENT_RESOURCE_WORKLOAD, ()), #R1
        # (feature_factory.EVENT_CURRENT_TOTAL_WORKLOAD, ()), #R2
        # (feature_factory.EVENT_RESOURCE, ()), #R3
        (feature_factory.EVENT_EXECUTION_DURATION, ()),  # P1
        (feature_factory.EVENT_ELAPSED_TIME, ()),  # P2
        TARGET_LABEL,  # P3
        (feature_factory.EVENT_FLOW_TIME, ()),  # P4
        (feature_factory.EVENT_SYNCHRONIZATION_TIME, ()),  # P5
        (feature_factory.EVENT_SOJOURN_TIME, ()),  # P6
        # (feature_factory.EVENT_POOLING_TIME, ()), #P7
        # (feature_factory.EVENT_LAGGING_TIME, ()), #P8
        # (feature_factory.EVENT_SERVICE_TIME, ()),  # P9
        # (feature_factory.EVENT_WAITING_TIME, ()), #P10
        # (feature_factory.EVENT_CURRENT_TOTAL_OBJECT_COUNT, ()), #O1
        (feature_factory.EVENT_PREVIOUS_OBJECT_COUNT, ()),  # O2
        (feature_factory.EVENT_PREVIOUS_TYPE_COUNT, ("GDSRCPT",)),  # O3
        # (feature_factory.EVENT_OBJECTS, ()), #O4
        (feature_factory.EVENT_NUM_OF_OBJECTS, ()),  # O5
    ]
)
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
