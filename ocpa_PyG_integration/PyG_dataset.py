# import pandas as pd
from tqdm import tqdm
import os

from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage
import pickle
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

"""
!!!
NOTE: This file was replaced by dataset_featurizer.py
but is kept to illustrate how to build a custom dataset in PyG.
!!!
"""


class EventGraphDataset(Dataset):
    """
    Class that serves as an interface between ocpa and PyG.

    Specifically, it imports from a Feature_Storage class and works with PyG for implementing a GNN.
    """

    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.test = test
        self.filename = filename
        super(EventGraphDataset, self).__init__(root, transform, pre_transform)
        print(f"TRACE: EventGraphDataset.__init__(root={root}, filename={filename})")

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        (The download func. is not implemented here)
        """
        print("TRACE: EventGraphDataset.raw_file_names")
        return self.filename

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""

        # Added:
        # print(os.getcwd())
        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)

        # Removed:
        # self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        # Changed:
        print("TRACE: EventGraphDataset.processed_file_names")
        if self.test:
            return [f"data_test_{i}.pt" for i in range(len(self.data.feature_graphs))]
        else:
            return [f"data_{i}.pt" for i in range(len(self.data.feature_graphs))]

    # def download(self):
    #     pass

    def process(self):
        """Processes a Feature_Storage object into PyG instance graph objects"""
        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)

        for index, feature_graph in tqdm(enumerate(self.data.feature_graphs)):
            # Split off labels from nodes,
            # and return full graph (cleansed of labels), and list of labels
            labels = self._split_X_y(feature_graph, ("event_remaining_time", ()))
            # Get node features
            node_feats = self._get_node_features(feature_graph)
            # Get edge features
            edge_feats = self._get_edge_features(feature_graph)
            # Get adjacency matrix
            edge_index = self._get_adjacency_matrix(feature_graph)

            # Create data object
            data = Data(
                x=node_feats,
                edge_index=edge_index,
                edge_attr=edge_feats,
                y=labels,
            )
            if self.test:
                torch.save(
                    data, os.path.join(self.processed_dir, f"data_test_{index}.pt")
                )
            else:
                torch.save(data, os.path.join(self.processed_dir, f"data_{index}.pt"))

    def _split_X_y(
        self, feature_graph: FeatureStorage.Feature_Graph, label_key
    ) -> list[torch.float]:
        """
        Impure function that splits off the target label from a feature graph
        and returns them both separately in a tuple of shape
        [A Feature_Graph, Number of Nodes]

        NOTE: This function can only be called once, since after it the label
        key is not present anymore, resulting in a KeyError.
        Also, it should be executed first in the processing pipeline.
        """
        ys = [node.attributes.pop(label_key) for node in feature_graph.nodes]

        return torch.tensor(ys, dtype=torch.float)

    def _get_node_features(
        self, feature_graph: FeatureStorage.Feature_Graph
    ) -> torch.Tensor:
        """
        This will return a feature matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        node_feature_matrix: list[list[torch.float]] = []

        for node in feature_graph.nodes:
            node_feats = list(node.attributes.values())
            # Append node features to matrix
            node_feature_matrix.append(node_feats)

        node_feature_matrix = np.asarray(node_feature_matrix)
        return torch.tensor(node_feature_matrix, dtype=torch.float)

    def _get_edge_features(self, mol):
        """
        This will return a matrix / 2d array of the shape
        [Number of edges, Edge Feature size]
        """
        pass

    def _get_adjacency_matrix(
        self, feature_graph: FeatureStorage.Feature_Graph
    ) -> torch.Tensor:
        """
        Function that returns the directed adjacency matrix in COO format, given a graph
        [2, Number of edges]
        """
        # Map event_id to node_index (counting from 0) using a dictionary
        node_index_map = {
            id: i
            for i, id in enumerate([node.event_id for node in feature_graph.nodes])
        }
        # Actually map event_id to node_index
        # so we have an index-based (event_id-agnostic) directed COO adjacency_matrix.
        adjacency_matrix_COO = [
            [node_index_map[e.source] for e in feature_graph.edges],
            [node_index_map[e.target] for e in feature_graph.edges],
        ]

        return torch.tensor(adjacency_matrix_COO, dtype=torch.long)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """- Equivalent to __getitem__ in pytorch
        - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f"data_test_{idx}.pt"))
        else:
            data = torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
        return data
