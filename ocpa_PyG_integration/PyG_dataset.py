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
    Class that serves as an adapter/bridge between ocpa and PyG.

    Specifically, it imports from a Feature_Storage class and works with PyG for implementing a GNN.
    """

    def __init__(
        self,
        root,
        filename,
        label_key: str,
        train: bool = False,
        test: bool = False,
        verbosity: int = 1,
        transform=None,
        pre_transform=None,
    ):
        """
        root (string, optional): Where the dataset should be stored. This folder is split
            into raw_dir (downloaded dataset) and processed_dir (processed data).

        train (bool, optional): If this is set the train indices Feature_Storage will be used.
            Use this when constructing the train split of the data set.
            If both train and test are not set, the whole Feature_Storage will
            be used as a data set (not recommended).

        test (bool, optional): If this is set the test indices of Feature_Storage will be used.
            Use this when constructing the test split of the data set.
            If both train and test are not set, the whole Feature_Storage will
            be used as a data set (not recommended).
        """
        self.label_key = label_key
        self.train = train
        self.test = test
        self.filename = filename
        self._verbosity = verbosity
        super(EventGraphDataset, self).__init__(root, transform, pre_transform)

        # print(f"TRACE: EventGraphDataset.__init__(root={root}, filename={filename})")

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        (The download func. is not implemented here)
        """
        # print("TRACE: EventGraphDataset.raw_file_names")
        return self.filename

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""

        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)

        if self.train:
            return [f"data_train_{i}.pt" for i in self.data.training_indices]
        if self.test:
            return [f"data_test_{i}.pt" for i in self.data.test_indices]
        else:
            return [f"data_{i}.pt" for i in range(len(self.data.feature_graphs))]

    # def download(self):
    #     pass

    def process(self):
        """Processes a Feature_Storage object into PyG instance graph objects"""

        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)

        # if train-test split is defined:
        #   process graph instances with a train or test tag.
        # else:
        #   do not give a tag to files that are being saved to disk

        if self.train:
            # Retrieve graphs with train indices
            train_graphs = [
                self.data.feature_graphs[i] for i in self.data.training_indices
            ]
            # Set Dataset size
            self._set_size(len(train_graphs))
            # Save each graph instance
            for index, feature_graph in self.__custom_verbosity_enumerate(
                train_graphs, self._verbosity
            ):
                self._write_graph_to_disk(
                    graph=feature_graph,
                    filename=f"data_train_{index}.pt",
                )
        elif self.test:
            # Retrieve graphs with test indices
            test_graphs = [self.data.feature_graphs[i] for i in self.data.test_indices]
            # Set Dataset size
            self._set_size(len(test_graphs))
            # Save each graph instance
            for index, feature_graph in self.__custom_verbosity_enumerate(
                test_graphs, self._verbosity
            ):
                self._write_graph_to_disk(
                    graph=feature_graph,
                    filename=f"data_test_{index}.pt",
                )
        else:
            # Save each graph instance
            for index, feature_graph in self.__custom_verbosity_enumerate(
                self.data.feature_graphs, self._verbosity
            ):
                self._write_graph_to_disk(
                    graph=feature_graph,
                    filename=f"data_{index}.pt",
                )

    def _write_graph_to_disk(
        self,
        graph: FeatureStorage.Feature_Graph,
        filename: str,
        # label_key: Any,
        # test: bool,
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
        if self.train or self.test:
            return self._size
        elif len(self.data.feature_graphs) == self._size:
            raise Exception(
                "Total number of graphs in Feature_Storage is equal to EventGraphDataset._size, which should be equal to either the train or test-set size"
            )
        else:
            return len(self.data.feature_graphs)

    def get(self, idx):
        """- Equivalent to __getitem__ in pytorch
        - Is not needed for PyG's InMemoryDataset
        """
        if self.train:
            return torch.load(os.path.join(self.processed_dir, f"data_train_{idx}.pt"))
        if self.test:
            return torch.load(os.path.join(self.processed_dir, f"data_test_{idx}.pt"))
        else:
            return torch.load(os.path.join(self.processed_dir, f"data_{idx}.pt"))
