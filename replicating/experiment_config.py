import random

import numpy as np
import ocpa.algo.predictive_monitoring.factory as feature_factory
import torch

STORAGE_PATH = "data/ocpa-processed"
FEATURE_STORAGE_FILE = "BPI17-scaled-split.fs"
RANDOM_SEED = 42
TARGET_LABEL = (feature_factory.EVENT_REMAINING_TIME, ())
SUBGRAPH_SIZE = 12
EPOCHS = 30
BATCH_SIZE = 512

# Initializing random seeds for maximizing reproducibility
torch.manual_seed(RANDOM_SEED)
generator = torch.Generator().manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# torch.use_deterministic_algorithms(True) # incompatible with GCN


def seed_worker(worker_id) -> None:
    worker_seed = torch.initial_seed() % RANDOM_SEED
    # worker_seed = RANDOM_SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)
