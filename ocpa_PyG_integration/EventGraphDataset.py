# import pandas as pd
from tqdm import tqdm
import os
from typing import Any
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage
import pickle
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class EventGraphDataset(Dataset):
    """
    Class that serves as an adapter between ocpa and PyG.

    Specifically, it imports from a Feature_Storage class and works with PyG for implementing a GNN.

    TODO:
    - Add possibility to load Feature_Storage object from memory, instead of pickled file.
    """

    def __init__(
        self,
        root,
        filename,
        label_key: str,
        train: bool = False,
        validation: bool = False,
        test: bool = False,
        verbosity: int = 1,
        transform=None,
        pre_transform=None,
    ):
        """
        root (string, optional): Where the dataset should be stored. This folder is split
            into raw_dir (downloaded dataset) and processed_dir (processed data).

        train (bool, optional): If True, train indices of Feature_Storage will be used.
            Use this when constructing the train split of the data set.
            If train, validation, and test are all False, the whole Feature_Storage will
            be used as a data set (not recommended).

        validation (bool, optional): If True, validation indices of Feature_Storage will be used.
            Use this when constructing the validation split of the data set.
            If train, validation, and test are all False, the whole Feature_Storage will
            be used as a data set (not recommended).

        test (bool, optional): If True, test indices of Feature_Storage will be used.
            Use this when constructing the test split of the data set.
            If train, validation, and test are all False, the whole Feature_Storage will
            be used as a data set (not recommended).
        """
        self.label_key = label_key
        self.train = train
        self.validation = validation
        self.test = test
        self.filename = filename
        self._verbosity = verbosity
        # Set filename according to type of dataset
        self._base_filename = "data"
        if self.train:
            self._base_filename += "_train"
        elif self.validation:
            self._base_filename += "_val"
        elif self.test:
            self._base_filename += "_test"
        super(EventGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""

        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)

        if self.train:
            return [f"{self._base_filename}_{i}.pt" for i in self.data.train_indices]
        if self.validation:
            return [
                f"{self._base_filename}_{i}.pt" for i in self.data.validation_indices
            ]
        if self.test:
            return [f"{self._base_filename}_{i}.pt" for i in self.data.test_indices]
        else:
            return [
                f"{self._base_filename}_{i}.pt"
                for i in range(len(self.data.feature_graphs))
            ]

    def process(self):
        """Processes a Feature_Storage object into PyG instance graph objects"""

        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)

        if self.train:
            # Retrieve graphs with train indices and write to disk
            self._graphs_to_disk(
                [self.data.feature_graphs[i] for i in self.data.train_indices]
            )
        elif self.validation:
            # Retrieve graphs with validation indices and write to disk
            self._graphs_to_disk(
                [self.data.feature_graphs[i] for i in self.data.validation_indices]
            )
        elif self.test:
            # Retrieve graphs with test indices and write to disk
            self._graphs_to_disk(
                [self.data.feature_graphs[i] for i in self.data.test_indices]
            )
        else:
            # Write all graphs to disk
            self._graphs_to_disk(self.data.feature_graphs)

    def _graphs_to_disk(
        self,
        graphs: list[FeatureStorage.Feature_Graph],
    ):
        # Set dataset size
        self._set_size(len(graphs))
        # Save each graph instance
        for index, feature_graph in self.__custom_verbosity_enumerate(
            graphs, self._verbosity
        ):
            self._one_graph_to_disk(
                graph=feature_graph,
                filename=f"{self._base_filename}_{index}.pt",
            )

    def _one_graph_to_disk(
        self,
        graph: FeatureStorage.Feature_Graph,
        filename: str,
    ):
        # Split off labels from nodes,
        # and return full graph (cleansed of labels), and list of labels
        labels = self._split_X_y(graph, self.label_key)

        # Get node features
        node_feats = self._get_node_features(graph)

        # Get edge features
        # edge_feats = self._get_edge_features(feature_graph)

        # Get adjacency matrix
        edge_index = self._get_adjacency_matrix(graph)

        # Create data object
        data = Data(
            y=labels,
            x=node_feats,
            edge_index=edge_index,
            # edge_attr=edge_feats,
        )

        torch.save(data, os.path.join(self.processed_dir, filename))

    def _split_X_y(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        label_key: Any,
    ) -> list[torch.float]:
        """
        Impure function that splits off the target label from a feature graph
        and returns them both separately in a tuple of shape
        [A Feature_Graph, Number of Nodes]

        NOTE: This function should only be called once, since after it the label
        key is not present anymore, resulting in a KeyError.
        Also, it should be executed first in the processing pipeline (i.e. in self.process()).
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
        # Append node features to matrix
        node_feature_matrix: list[list[torch.float]] = [
            list(node.attributes.values()) for node in feature_graph.nodes
        ]

        return torch.tensor(node_feature_matrix, dtype=torch.float)

    def _get_edge_features(self, feature_graph: FeatureStorage.Feature_Graph):
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

    def _set_size(self, size: int) -> None:
        """Set the number of graphs stored in this EventGraphDataset object."""
        self._size = size

    def __custom_verbosity_enumerate(self, iterable, miniters: int):
        """Return either just the enumerated iterable, or one with the progress tracked."""
        if self._verbosity:
            return tqdm(enumerate(iterable), miniters=miniters)
        else:
            return enumerate(iterable)

    def len(self) -> int:
        if self.train or self.test or self.validation:
            if len(self.data.feature_graphs) == self._size:
                raise Exception(
                    "Total number of graphs in Feature_Storage is equal to EventGraphDataset._size, but the latter should be equal to either the train-, validation-, or test-set size"
                )
            return self._size
        else:
            return len(self.data.feature_graphs)

    def get(self, idx):
        """
        - Equivalent to __getitem__ in PyTorch
        - Is not needed for PyG's InMemoryDataset
        """
        return torch.load(
            os.path.join(self.processed_dir, f"{self._base_filename}_{idx}.pt")
        )
