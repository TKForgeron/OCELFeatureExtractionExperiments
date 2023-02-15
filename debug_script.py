from ocpa_PyG_integration.PyG_dataset import EventGraphDataset

ds_train = EventGraphDataset(
    root="data/ocpa-processed/",
    filename="BPI2017-feature_storage_split.pkl",
    label_key=("event_remaining_time", ()),
    train=True,
)

ds_test = EventGraphDataset(
    root="data/ocpa-processed/",
    filename="BPI2017-feature_storage_split.pkl",
    label_key=("event_remaining_time", ()),
    test=True,
)
