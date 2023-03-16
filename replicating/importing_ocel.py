# SELF REMINDER. Copy the 'ocpa' directory to the forked one from github, so that I can push updates to github.
# Timing
import timeit

start = timeit.default_timer()

# Python natives
import pickle

# Object centric process mining
import ocpa.objects.log.importer.csv.factory as csv_import_factory
import ocpa.algo.predictive_monitoring.factory as feature_factory

# Simple machine learning models, procedure tools, and evaluation metrics
from sklearn.preprocessing import StandardScaler, PowerTransformer

# Global configuration
from experiment_config import (
    STORAGE_PATH,
    FEATURE_STORAGE_FILE,
    RANDOM_SEED,
    TARGET_LABEL,
)


def print_time_taken(start_time: float, prefix: str = "") -> float:
    elapsed_time = timeit.default_timer() - start_time
    minutes, seconds = int(elapsed_time / 60), int(elapsed_time % 60)
    print(f"{prefix} time taken: {minutes}m{seconds}s")


filename = "data/adams/example_logs/mdl/BPI2017-Final.csv"
object_types = ["application", "offer"]
parameters = {
    "obj_names": object_types,
    "val_names": ["event_RequestedAmount"],
    "act_name": "event_activity",
    "time_name": "event_timestamp",
    "sep": ",",
}
print("Constructing OCEL object")
ocel = csv_import_factory.apply(filename, csv_import_factory.TO_OCEL, parameters)

activities = ocel.log.log["event_activity"].unique().tolist()

feature_set = [
    (feature_factory.EVENT_PRECEDING_ACTIVITIES, (act,)) for act in activities
] + [  # C2
    (
        feature_factory.EVENT_AGG_PREVIOUS_CHAR_VALUES,
        ("event_RequestedAmount", max),
    ),  # D1
    (feature_factory.EVENT_ELAPSED_TIME, ()),  # P2
    TARGET_LABEL,  # P3
    (feature_factory.EVENT_PREVIOUS_TYPE_COUNT, ("offer",)),  # O3
]
print("Constructing FeatureStorage object")
feature_storage = feature_factory.apply(ocel, event_based_features=feature_set)

print("Pickling FeatureStorage object")
with open(
    f"{STORAGE_PATH}/raw/BPI17-feature_storage-[C2,D1,P2,P3,O3].fs",
    "wb",
) as file:
    pickle.dump(feature_storage, file)

print_time_taken(start, "Total")
