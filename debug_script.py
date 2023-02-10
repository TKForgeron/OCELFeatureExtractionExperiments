# import os

# os.chdir("c:\\Users\\Tim\\Development\\OCELFeatureExtractionExperiments\\")
# print(os.getcwd())
# Py Geometric
from ocpa_PyG_integration.PyG_dataset import EventGraphDataset

ds = EventGraphDataset(
    root="data/ocpa-processed/",
    filename="BPI2017-feature_storage_split.pkl",
)
