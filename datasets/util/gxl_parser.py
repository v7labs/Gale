import os
import xml.etree.ElementTree as ET
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from datasets.util.custom_exception import InvalidFileException
import logging


class ParsedGxlDataset:
    def __init__(self, path_to_dataset, categorical_features=None, subset=None, remove_coordinates=False,
                 features_to_use=None):
        """
        This class creates a dataset object containing all the graphs parsed from gxl files as ParsedGxlGraph objects

        Parameters
        ----------
        path_to_dataset: str
            path to the 'data' folder with gxl files
        categorical_features : dict
            optional parameter: dictionary with first level 'edge' and/or 'node' and then a dict of dict(s) with the
            name of the attribute and a dict of its respective one-hot encoding
            e.g. {'node': {'symbol': {'C': [1, 0, 0], 'H': [0, 1, 0], 'O: [0, 0, 1]}}}
        subset: if specified, has to be either 'train', 'test' or 'val'
        """
        self.path_to_dataset = path_to_dataset

        # optional arguments
        self.categorical_features = categorical_features
        self.features_to_use = features_to_use
        self.remove_coordinates = remove_coordinates
        self.center_coordinates = center_coordinates
        self.subset = subset

    @property
    def path_to_dataset(self) -> str:
        return self._path_to_dataset

    @path_to_dataset.setter
    def path_to_dataset(self, path_to_dataset) -> str:
        if not os.path.isdir(path_to_dataset):
            logging.error(f'Folder {path_to_dataset} does not exist.')
            sys.exit(-1)
        self._path_to_dataset = path_to_dataset

    @property
    def subset(self) -> str:
        return self._subset

    @subset.setter
    def subset(self, subset) -> str:
        # if set ensure that it is a valid arguments
        if subset and subset not in ['train', 'val', 'test']:
            logging.error(f"Subset has to be specified as either 'train', 'val' or 'test'")
            sys.exit(-1)
        self._subset = subset

    @property
    def edge_mode(self) -> str:
        return self.graphs[0].edgemode

    def config(self) -> dict:
        config = {
            'dataset_name': self.name,
            'categorical_features': self.categorical_features,
            'node_feature_names': self.node_feature_names,
            'edge_feature_names': self.edge_feature_names,
            'datset_split': self.dataset_split,
            'class_encoding': self.class_int_encoding,
            'classes': self.classes
        }
        if hasattr(self, 'nodes_onehot'):
            config['one-hot_encoding_node_features'] = self.nodes_onehot
        if hasattr(self, 'edges_onehot'):
            config['one-hot_encoding_edge_features'] = self.edges_onehot

        return config

    def get_dataset_split(self) -> dict:
        """
        Create a dictionary that contains the dataset split as well as the class labels

        Returns
        -------
        filename_class_split: dict
            {'train': {'file1.gxl': 'class_label' }, ...}
        """
        filename_class_split = {}

        for subset in ['train', 'test', 'val']:
            cxl_file = os.path.join(self.root_path, subset + '.cxl')
            if not os.path.isfile(os.path.join(self.root_path, subset + '.cxl')):
                logging.error(f'File {cxl_file} not found. Make sure file is called either train.cxl, val.cxl or test.cxl')
            tree = ET.parse(cxl_file)
            root = tree.getroot()
            filename_class_split[subset] = {i.attrib['file']: i.attrib['class'] for i in root.iter('print')}

        return filename_class_split

    def __len__(self) -> int:
        return len(self.graphs)


class ParsedGxlGraph:
    def __init__(self, path_to_gxl, subset, class_label, remove_coordinates=False, features_to_use=None):
        """
        This class contains all the information encoded in a single gxl file = one graph
        Parameters
        ----------
        subset : str
            either 'test', 'val' or 'train'
        class_label : str
            class label of the graph
        path_to_gxl: str
            path to the gxl file
        """
        self.filepath = path_to_gxl
        self.subset = subset
        self.class_label = class_label
        self.remove_coordinates = remove_coordinates
        features_to_use = features_to_use

        # name of the gxl file (without the ending)
        self.filename = os.path.basename(self.filepath).split('.')[0]

        # parsing the gxl
        # sets up the following properties: node_features, node_feature_names, edges, edge_features, edge_feature_names,
        # node_position, graph_id, edge_ids_present and edgemode
        self.setup_graph_features()

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, path_to_gxl) -> str:
        if not os.path.isfile(path_to_gxl):
            logging.error(f'File {path_to_gxl} does not exist.')
            sys.exit(-1)
        self._filepath = path_to_gxl

    @property
    def subset(self) -> str:
        return self._subset

    @subset.setter
    def subset(self, subset) -> str:
        # if set ensure that it is a valid arguments
        if subset and subset not in ['train', 'val', 'test']:
            logging.error(f"Subset has to be specified as either 'train', 'val' or 'test'")
            sys.exit(-1)
        self._subset = subset

    def setup_graph_features(self):
        """
        Parses the gxl file and sets the following graph properties
        - graph info: graph_id, edge_ids_present and edgemode
        - node: node_features, node_feature_names, and node_position
        - edge: edges, edge_features and edge_feature_names

        Returns
        -------
        self.get_graph_attr(root), node_feature_names, node_features, node_position, edge_feature_names, edge_features, edges:
            ( (str, bool, str), [str], [mixed], [float / int], [str], [mixed], [[int, int]] )
        """
        tree = ET.parse(self.filepath)
        root = tree.getroot()

        # verify that the file contains the expected attributes (node, edge, id, edgeids and edgemode)
        self.sanity_check(root)

        self.edges = self.get_edges(root)
        self.node_feature_names, self.node_features = self.get_features(root, 'node')

        # remove the x and y node features and put them in their own variable
        if self.node_feature_names is not None and 'x' in self.node_feature_names and 'y' in self.node_feature_names:
            x_ind = self.node_feature_names.index('x')
            y_ind = self.node_feature_names.index('y')
            self.node_position = [[node[x_ind], node[y_ind]] for node in self.node_features]
            # remove the positions from the node features if we don't want to use them
            if self.remove_coordinates:
                for node in self.node_features:
                    del node[x_ind]
                    del node[y_ind-1]
                del self.node_feature_names[x_ind]
                del self.node_feature_names[y_ind-1]
        else:
            self.node_position = []

        self.edge_feature_names, self.edge_features = self.get_features(root, 'edge')
        self.graph_id, self.edge_ids_present, self.edgemode = self.get_graph_attr(root)

    def sanity_check(self, root):
        """
        Check if files contain the expected content

        Parameters
        ----------
        root:

        Returns
        -------
        None
        """
        # check if node, edge, edgeid, edgemode, edgemode keyword exists
        if len([i.attrib for i in root.iter('graph')][0]) != 3:
            raise InvalidFileException

        if len([node for node in root.iter('node')]) == 0:
            logging.warning(f'File {self.filepath} is an empty graph!')
        elif len([edge for edge in root.iter('edge')]) == 0:
            logging.warning(f'File {self.filepath} has no edges!')

    def normalize(self, mean_std, center_coordinates):
        """
        This method normalizes the node and edge features (if present) and initializes a random node for empty graphs

        Parameters
        ----------
        graph: ParsedGxlGraph

        mean_std: dict
            dictionary containing the mean and standard deviation for the node and edge features
        center_coordinates: set if coordinates need to be centered (xy-vector - xy(average)-vector)

        Returns
        ----------
        Normalized graph

        """
        # TODO: also center the coordinates, if they are used
        # TODO: make this work for when only selected features are used
        def z_normalize(feature, mean, std):
            return (feature - mean) / std

        node_mean = mean_std['node_features']['mean']
        node_std = mean_std['node_features']['std']
        edge_mean = mean_std['edge_features']['mean']
        edge_std = mean_std['edge_features']['std']

        # normalize the node features, if none are present initialize a random node
        # if len(graph.node_features) > 0:
        #     for node_ind in range(len(graph.node_features)):
        #         graph.node_features[node_ind] = [z_normalize(graph.node_features[node_ind][i], node_mean[i], node_std[i])
        #                                          for i in range(len(node_mean))]
        # else:
        #    graph.node_features = [np.random.normal(node_mean, node_std, len(node_mean))]

        # check if graph is not empty
        if len(graph.node_features) > 0:
            # normalize the node features
            for node_ind in range(len(graph.node_features)):
                graph.node_features[node_ind] = [
                    z_normalize(graph.node_features[node_ind][i], node_mean[i], node_std[i])
                    for i in range(len(node_mean))]
            # normalize the edge features
            for edge_ind in range(len(graph.edge_features)):
                graph.edge_features[edge_ind] = [
                    z_normalize(graph.edge_features[edge_ind][i], edge_mean[i], edge_std[i])
                    for i in range(len(edge_mean))]

        return graph

    @staticmethod
    def get_graph_attr(root) -> tuple:
        """
        Gets the information attributes of the whole graph:
        Parameters
        ----------
        root: gxl element
            root of ET tree

        Returns
        -------
        tuple (str, bool, str)
            ID of the graph, Edge IDs present (true / false), edge mode (directed / undirected)
        """
        graph = [i.attrib for i in root.iter('graph')]
        assert len(graph) == 1
        g = graph[0]
        return g['id'], g['edgeids'] == 'True', g['edgemode']

    @staticmethod
    def get_edges(root) -> list:
        """
        Get the start and end points of every edge and store them in a list of lists (from the element tree, gxl)
        Parameters
        ----------
        root: gxl element

        Returns
        -------
        [[int, int]]
            list of indices of connected nodes
        """
        edge_list = []

        start_points = [int(edge.attrib["from"].replace('_', '')) for edge in root.iter('edge')]
        end_points = [int(edge.attrib["to"].replace('_', '')) for edge in root.iter('edge')]
        assert len(start_points) == len(end_points)

        # move enumeration start to 0 if necessary
        if len(start_points) > 0 and len(end_points) > 0:
            if min(min(start_points, end_points)) > 0:
                shift = min(min(start_points, end_points))
                start_points = [x - shift for x in start_points]
                end_points = [x - shift for x in end_points]

            edge_list = [[start_points[i], end_points[i]] for i in range(len(start_points))]

        return edge_list

    @staticmethod
    def decode_feature(f) -> str:
        data_types = {'string': str,
                      'float': float,
                      'int': int}

        # convert the feature value to the correct data type as specified in the gxl
        return data_types[f.tag](f.text.strip())