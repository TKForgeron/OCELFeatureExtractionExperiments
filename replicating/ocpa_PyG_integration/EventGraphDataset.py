# import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
import os
from ocpa.algo.predictive_monitoring.obj import Feature_Storage as FeatureStorage
import pickle
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from warnings import warn

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


@dataclass
class SubGraphParameters:
    """
    Class that holds information on how the Feature_Graphs are subgraph-sampled.

    If subgraph sampling is not used in EventGraphDataset,
     then 'actual_subgraph_sampling' is set to 'False'
     and 'graph_indices' keeps a list of the full graphs that were saved.
    """

    # __slots__ = "size", "graph_subgraph_index_map"
    size: int
    path: str
    actual_subgraph_sampling: bool = field(init=False)
    graph_indices: set[int] = field(default_factory=set)
    graph_subgraph_index_map: dict[int, list[int]] = field(default_factory=dict)

    def __post_init__(self):
        if self.size:
            self.actual_subgraph_sampling = True
        else:
            self.actual_subgraph_sampling = False
        self.path = self.path

    def add_subgraph(self, graph_idx: int, subgraph_idx: int) -> None:
        if self.actual_subgraph_sampling:
            if graph_idx in self.graph_subgraph_index_map:
                self.graph_subgraph_index_map[graph_idx].append(subgraph_idx)
            else:
                self.graph_subgraph_index_map[graph_idx] = [subgraph_idx]
        else:
            warn(
                f"You are trying to add a subgraph, while `SubGraphParameters.actual_subgraph_sampling` is set to '{self.actual_subgraph_sampling}'"
            )

    def add_graph(self, graph_idx: int) -> None:
        if not self.actual_subgraph_sampling:
            self.graph_indices.add(graph_idx)
        else:
            warn(
                f"You are trying to add a full graph, while `SubGraphParameters.actual_subgraph_sampling` is set to '{self.actual_subgraph_sampling}'"
            )


class EventGraphDataset(Dataset):
    """
    Class that serves as an adapter between ocpa and PyG.

    Specifically, it imports from a Feature_Storage class and works with PyG for implementing a GNN.

    TODO:
    - add if statements handling subgraph samplingn naming convention in: processed_file_names() and get()
    - currently, we record y per node, maybe we should try saving one y per graph (only y for the last node in a graph)
    - add event indices as node indices, like in gnn_utils.py (generate_graph_dataset())
    - Add possibility to load Feature_Storage object from memory, instead of pickled file.
    """

    def __init__(
        self,
        root,
        filename,
        label_key: str,
        size_subgraph_samples: int = None,
        train: bool = False,
        validation: bool = False,
        test: bool = False,
        verbosity: int = 1,
        transform=None,
        pre_transform=None,
        file_extension: str = "pt",
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

        NOTE: For disambiguation purposes, througout this class, a distinction
            has been made between 'graph' and 'feature graph'.
            The first being of class `torch_geometric.data.Data` and the latter being
            of class `ocpa.algo.predictive_monitoring.obj.Feature_Graph.Feature_Storage`
        """
        self.filename = filename
        self.label_key = label_key
        self.subgraph_params = SubGraphParameters(size=size_subgraph_samples, path=root)
        self.train = train
        self.validation = validation
        self.test = test
        self._verbosity = verbosity
        self._base_filename = "data"
        if self.train:
            self._base_filename += "_train"
        elif self.validation:
            self._base_filename += "_val"
        elif self.test:
            self._base_filename += "_test"
        self._file_extension = file_extension
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

        # with open(self.raw_paths[0], "rb") as file:
        #     self.data = pickle.load(file)
        try:
            with open(
                os.path.join(self.processed_dir, "subgraph_parameters.pt"), "rb"
            ) as file:
                self.subgraph_params = pickle.load(file)
        except:
            # the dataset has not (or not with the same settings) run before
            # so return an empty list, and don't skip processing
            return []

        if self.subgraph_params.actual_subgraph_sampling:
            get_filestring = (
                lambda g_id, sg_id: f"{self._base_filename}_{g_id}_{sg_id}.{self._file_extension}"
            )
            map_items = self.subgraph_params.graph_subgraph_index_map.items()
            flatten = lambda l: [item for sublist in l for item in sublist]

            return flatten(
                [
                    [
                        get_filestring(mapping[0], subgraph_idxs)
                        for subgraph_idxs in mapping[1]
                    ]
                    for mapping in map_items
                ]
            )
        else:
            return [
                f"{self._base_filename}_{graph_idx}.{self._file_extension}"
                for graph_idx in self.subgraph_params.graph_indices
            ]

    def _set_size(self, size: int) -> None:
        """Sets the number of graphs stored in this EventGraphDataset object."""
        self._size = size

    def _get_size(self) -> int:
        """Gets the number of graphs stored in this EventGraphDataset object."""
        return self._size

    size: int = property(_get_size, _set_size)

    def process(self):
        """Processes a Feature_Storage object into PyG instance graph objects"""

        # Retrieve Feature_Storage object from disk
        with open(self.raw_paths[0], "rb") as file:
            self.data = pickle.load(file)

        if self.train:
            # Retrieve feature graphs with train indices and write to disk
            self._feature_graphs_to_disk(
                [self.data.feature_graphs[i] for i in self.data.train_indices]
            )
        elif self.validation:
            # Retrieve graphs with validation indices and write to disk
            self._feature_graphs_to_disk(
                [self.data.feature_graphs[i] for i in self.data.validation_indices]
            )
        elif self.test:
            # Retrieve graphs with test indices and write to disk
            self._feature_graphs_to_disk(
                [self.data.feature_graphs[i] for i in self.data.test_indices]
            )
        else:
            # Write all graphs to disk
            self._feature_graphs_to_disk(self.data.feature_graphs)

        with open(
            os.path.join(self.processed_dir, "subgraph_parameters.pt"), "wb"
        ) as file:
            pickle.dump(self.subgraph_params, file)

    def _feature_graphs_to_disk(
        self,
        feature_graphs: list[FeatureStorage.Feature_Graph],
    ):
        # Save each feature_graph instance
        for index, feature_graph in self.__custom_verbosity_enumerate(
            feature_graphs, miniters=self._verbosity
        ):
            # Save a feature_graph instance

            self._feature_graph_to_graph_to_disk(
                feature_graph=feature_graph,
                graph_idx=index,
            )
        # Set size of the dataset (this depends on whether subgraphs were sampled)
        if self.subgraph_params.actual_subgraph_sampling:
            self.size = sum(
                [
                    len(subgraph_idxs)
                    for subgraph_idxs in self.subgraph_params.graph_subgraph_index_map.values()
                ]
            )
        else:
            self.size = len(self.subgraph_params.graph_indices)

    def _feature_graph_to_graph_to_disk(
        self, feature_graph: FeatureStorage.Feature_Graph, graph_idx: int
    ) -> None:
        """
        Saves a FeatureStorage.Feature_Graph object as PyG Data object(s) to disk.

        Returns amount of PyG Data object instances that are saved, which depends
         on whether subgraphs will be sampled, and if yes, how large they will be
         (explicated in: EventGraphDataset.subgraph_params.size)
        """
        # Split off labels from nodes,
        # and return full graph (cleansed of labels), and list of labels
        labels = self._split_X_y(feature_graph, self.label_key)
        #  np.argsort([node.event_id for node in graph.nodes])[-1]

        # Get node features
        node_feats = self._get_node_features(feature_graph)

        # Get edge features
        # edge_feats = self._get_edge_features(feature_graph)

        # Get adjacency matrix
        edge_index = self._get_adjacency_matrix(feature_graph)

        # Create graph data object
        data = Data(
            y=labels,
            x=node_feats,
            edge_index=edge_index,
            # edge_attr=edge_feats,
        )

        if self.subgraph_params.size:
            # Retrieve indices that would sort the nodes in the graph
            sorted_node_indices = np.argsort(
                [
                    self.__get_node_index_mapping(feature_graph)[node.event_id]
                    for node in feature_graph.nodes
                ]
            )
            # extract subgraph and label for each node set as terminal node
            k = self.subgraph_params.size
            if len(sorted_node_indices) != 0:
                for i in range(k - 1, len(sorted_node_indices)):
                    subgraph_idx = i - (k - 1)
                    subgraph = data.subgraph(
                        subset=torch.tensor(
                            range(subgraph_idx, i + 1), dtype=torch.long
                        )
                    )  # include last event
                    # subgraph_label = subgraph.ndata["remaining_time"].numpy()[-1]
                    torch.save(
                        subgraph,
                        os.path.join(
                            self.processed_dir,
                            f"{self._base_filename}_{graph_idx}_{subgraph_idx}.{self._file_extension}",
                        ),
                    )
                    # record naming scheme/graph indices, for loading/skipping processing later
                    self.subgraph_params.add_subgraph(graph_idx, subgraph_idx)

        else:
            torch.save(
                data,
                os.path.join(
                    self.processed_dir,
                    f"{self._base_filename}_{graph_idx}.{self._file_extension}",
                ),
            )
            # record naming scheme/graph indices, for loading/skipping processing later
            self.subgraph_params.add_graph(graph_idx)

    def _split_X_y(
        self,
        feature_graph: FeatureStorage.Feature_Graph,
        label_key: tuple[str, tuple],
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
        Returns the directed adjacency matrix in COO format, given a graph
        [2, Number of edges]
        """
        # Map event_id to node_index (counting from 0) using a dictionary
        node_index_map = self.__get_node_index_mapping(feature_graph)
        # Actually map event_id to node_index
        # so we have an index-based (event_id-agnostic) directed COO adjacency_matrix.
        adjacency_matrix_COO = [
            [node_index_map[e.source] for e in feature_graph.edges],
            [node_index_map[e.target] for e in feature_graph.edges],
        ]

        return torch.tensor(adjacency_matrix_COO, dtype=torch.long)

    def __get_node_index_mapping(
        self, feature_graph: FeatureStorage.Feature_Graph
    ) -> dict:
        """Returns a dictionary containing a mapping from event_ids to node indices in the given graph"""
        return {
            id: i
            for i, id in enumerate([node.event_id for node in feature_graph.nodes])
        }

    def __custom_verbosity_enumerate(self, iterable, miniters: int):
        """Returns either just the enumerated iterable, or one with the progress tracked."""
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

    def get(self, graph_idx):
        """
        - Equivalent to __getitem__ in PyTorch
        - Is not needed for PyG's InMemoryDataset
        """
        return torch.load(
            os.path.join(
                self.processed_dir,
                f"{self._base_filename}_{graph_idx}.{self._file_extension}",
            )
        )
