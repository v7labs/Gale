import os
import xml.etree.ElementTree as ET
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from datasets.util.custom_exception import InvalidFileException
import logging


class ParsedGxlDataset:
    def __init__(self, path_to_dataset: str, categorical_features: dict = None, subset: str = None,
                 remove_coordinates: bool = False, features_to_use: list = None) -> object:
        """
        This class creates a dataset object containing all the graphs parsed from gxl files as ParsedGxlGraph objects

        Parameters
        ----------
        remove_coordinates : bool
            if true, coordinates are removed from the node features
        path_to_dataset: str
            path to the 'data' folder with gxl files
        categorical_features : dict
            optional parameter: dictionary with first level 'edge' and/or 'node' and then a dict of dict(s) with the
            name of the attribute and a dict of its respective one-hot encoding
            e.g. {'node': {'symbol': {'C': [1, 0, 0], 'H': [0, 1, 0], 'O: [0, 0, 1]}}}
        subset: if specified, has to be either 'train', 'test' or 'val'
        """
        self.path_to_dataset = path_to_dataset
        self.name = os.path.basename(os.path.dirname(self.root_path))

        # optional arguments
        self.categorical_features = categorical_features
        self.features_to_use = features_to_use  # TODO: implement
        self.remove_coordinates = remove_coordinates
        self.subset = subset

        # get a list of the empty graphs
        self.invalid_files = []

        # get the node and edge features names at a higher level plus their data type
        # sets: node_feature_names, edge_feature_names, node_dtypes, and edge_dtypes
        self.set_feature_names_and_types()

        # if there are string-based features, we need to encode them as a one-hot vector
        if self.node_feature_names and (str in self.node_dtypes or len(self.categorical_features['node']) > 0):
            self.one_hot_encode_nodes()
        if self.edge_feature_names and (str in self.edge_dtypes or len(self.categorical_features['edge']) > 0):
            self.one_hot_encode_edge_features()

    def one_hot_encode_nodes(self):
        """
        TODO: continue here, simplify this
        This functions one-hot encodes the categorical node feature values and changes them in all the graphs
        (graph.node_features)

        Returns
        -------
        encoded_node_features: list
            one-hot encoding of the categorical node features
            [(feature index (str), feature name (str), encoding (dict)), (...), ... ]
        """
        name_to_ind = {name: ind for ind, name in enumerate(self.node_feature_names)}

        # get the name of the string features
        cat_node_features = list(set(self.categorical_features['node'] + [self.node_feature_names[index] for index in
                                                       [i for i, x in enumerate(self.node_dtypes) if x == str]]))
        if self.features_to_use:
            cat_node_features = [f for f in cat_node_features if f in self.features_to_use]

        self.categorical_features['node'] = cat_node_features

        encoded_node_features = []
        # get the one-hot encodings
        for feature_name in cat_node_features:
            # find the feature range and generate the one-hot encoding
            feature_ind = name_to_ind[feature_name]
            all_feature_values = list(
                set([edge[feature_ind] for graph in self.graphs for edge in graph.node_features]))
            encoded_node_features += self._get_encoding(feature_name, all_feature_values, name_to_ind)

        # overwrite the categorical node features with the argmax of the one-hot encoding
        for graph_ind in range(self.__len__()):
            for node_ind in range(self.graphs[graph_ind].nb_of_nodes):
                for feature_ind, name, encoding in encoded_node_features:
                    self.graphs[graph_ind].node_features[node_ind][feature_ind] = np.argmax(encoding[
                        self.graphs[graph_ind].node_features[node_ind][feature_ind]])

        self.nodes_onehot = encoded_node_features

    def one_hot_encode_edge_features(self):
        """
        TODO: continue here, simplify this
        This functions one-hot encodes the categorical edge feature values and changes them in all the
        graphs (graph.edge_features)

        Returns
        -------
        encoded_edge_features: list
            one-hot encoding of the categorical edge features
            [(feature index (str), feature name (str), encoding (dict)), (...), ... ]
        """
        name_to_ind = {name: ind for ind, name in enumerate(self.edge_feature_names)}

        # get the name of the string features
        cat_edge_features = list(set(self.categorical_features['edge'] + [self.edge_feature_names[index] for index in
                                                       [i for i, x in enumerate(self.edge_dtypes) if x == str]]))
        if self.features_to_use:
            cat_edge_features = [f for f in cat_edge_features if f in self.features_to_use]
        self.categorical_features['edge'] = cat_edge_features
        self.config['categorical_features'] = self.categorical_features

        # get the one-hot encodings
        encoded_edge_features = []
        for feature_name in cat_edge_features:
            # find the feature range and generate the one-hot encoding
            feature_ind = name_to_ind[feature_name]
            all_feature_values = list(
                set([edge[feature_ind] for graph in self.graphs for edge in graph.edge_features]))
            encoded_edge_features += self._get_encoding(feature_name, all_feature_values, name_to_ind)

        # overwrite the categorical edge features with the argmax of the one-hot encoding
        for graph_ind in range(self.__len__()):
            for edge_ind in range(self.graphs[graph_ind].nb_of_edges):
                for feature_ind, name, encoding in encoded_edge_features:
                    self.graphs[graph_ind].edge_features[edge_ind][feature_ind] = np.argmax(encoding[
                        self.graphs[graph_ind].edge_features[edge_ind][feature_ind]])

        self.nodes_onehot = encoded_edge_features

    @property
    def graphs(self) -> list:
        """
        Create the graph objects. If self.subset is set only the specified subset is loaded,
        otherwise the whole dataset is loaded.

        Returns
        -------
        graphs: list [ParsedGxlGraph obj]
            list of graphs parsed from the gxl files
        """
        graphs = []
        for filename in self.all_file_names:
            try:
                try:
                    subset, class_label = self.filename_split_class[filename]
                    if self.subset:
                        assert subset == self.subset
                except KeyError:
                    logging.warning(f'{filename} does not appear in the dataset split files. File is skipped.')
                    continue

                g = ParsedGxlGraph(os.path.join(self.path_to_dataset, filename), subset, self.class_int_encoding[class_label],
                                   remove_coordinates=self.remove_coordinates, features_to_use=self.features_to_use)
                graphs.append(g)
            except InvalidFileException:
                logging.warning(f'File {filename} is invalid. Please verify that the file contains the expected attributes '
                                f'(node, edge, id, edgeids and edgemode)')
                self.invalid_files.append(filename)

        return graphs

    @property
    def filename_split_class(self) -> dict:
        filename_split_class = {}
        if self.subset:
            for filename, class_label in self.dataset_split[self.subset].items():
                filename_split_class[filename] = (self.subset, class_label)
        else:
            for subset, d in self.dataset_split.items():
                for filename, class_label in d.items():
                    filename_split_class[filename] = (subset, class_label)
        return filename_split_class

    @property
    def class_names(self) -> list:
        return [c for c in sorted(set([i for i in self.filename_split_class.values()]))]

    @property
    def class_int_encoding(self) -> dict:
        return {c: i for i, c in enumerate(self.class_names())}

    @property
    def file_ids(self):
        # get a list of all the filenames without the path and the extension
        return [g.file_id for g in self.graphs]

    @property
    def all_file_names(self):
        if self.subset:
            filenames = [f for f in self.dataset_split[self.subset] if os.path.isfile(os.path.join(self.path_to_dataset, f)) if '.gxl' in f]
        else:
            filenames = [f for f in os.listdir(self.path_to_dataset) if os.path.isfile(os.path.join(self.path_to_dataset, f)) if '.gxl' in f]
        return filenames

    @property
    def path_to_dataset(self) -> str:
        return self._path_to_dataset

    @path_to_dataset.setter
    def path_to_dataset(self, path_to_dataset):
        if not os.path.isdir(path_to_dataset):
            logging.error(f'Folder {path_to_dataset} does not exist.')
            sys.exit(-1)
        self._path_to_dataset = path_to_dataset

    @property
    def subset(self) -> str:
        return self._subset

    @subset.setter
    def subset(self, subset):
        # if set ensure that it is a valid arguments
        if subset and subset not in ['train', 'val', 'test']:
            logging.error(f"Subset has to be specified as either 'train', 'val' or 'test'")
            sys.exit(-1)
        self._subset = subset

    @property
    def edge_mode(self) -> str:
        return self.graphs[0].edgemode

    @property
    def dataset_split(self) -> dict:
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

    @property
    def config(self) -> dict:
        # Setup the configuration dictionary
        config = {
            'dataset_name': self.name,
            'categorical_features': self.categorical_features,
            'node_feature_names': self.node_feature_names,
            'edge_feature_names': self.edge_feature_names,
            'datset_split': self.dataset_split,
            'class_encoding': self.class_int_encoding,
            'classes': self.class_names
        }
        if hasattr(self, 'nodes_onehot'):
            config['one-hot_encoding_node_features'] = self.nodes_onehot
        if hasattr(self, 'edges_onehot'):
            config['one-hot_encoding_edge_features'] = self.edges_onehot

        return config

    def set_feature_names_and_types(self):
        # get the node and edge feature names available at a higher level
        agraph = [g for g in self.graphs if len(g.node_features) > 0][0]
        self.node_feature_names = agraph.node_feature_names
        self.edge_feature_names = agraph.edge_feature_names

        # node / edge feature data type
        if len(agraph.node_features) > 0:
            self.node_dtypes = [type(dtype) for dtype in agraph.node_features[0]]
            assert len(self.node_feature_names) == len(self.node_dtypes)
        else:
            self.node_dtypes = None

        if self.edge_feature_names:
            self.edge_dtypes = [type(dtype) for dtype in agraph.edge_features[0]]
            assert len(self.edge_feature_names) == len(self.edge_dtypes)
        else:
            self.edge_feature_names = None

    @staticmethod
    def _get_encoding(feature_name, all_feature_values, name_to_ind):
        """
        Helper method to generate the one-hot encoding for the categorical features.

        Parameters
        ----------
        all_feature_values
        feature_name
        name_to_ind: dict
            contains the mapping of the feature name to its position in the feature vector

        Returns
        -------
        [(feature index (str), feature name (str), encoding (dict)), (...), ... ]
        """
        endoced_features = []

        # create the one-hot encoding
        integer_encoded = LabelEncoder().fit_transform(all_feature_values)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = OneHotEncoder(sparse=False).fit_transform(integer_encoded)

        # add the one-hot encoding to the dict
        endoced_features.append((name_to_ind[feature_name], feature_name,
                                      {all_feature_values[i]: encoding for i, encoding in enumerate(onehot_encoded)}))

        return endoced_features

    def __len__(self) -> int:
        return len(self.graphs)


class ParsedGxlGraph:
    def __init__(self, path_to_gxl: str, subset: str, class_label: int, remove_coordinates: bool = False,
                 features_to_use: dict = None) -> object:
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
        self.features_to_use = features_to_use  # TODO: implement

        # name of the gxl file (without the ending)
        self.file_id = os.path.basename(self.filepath).split('.')[0]

        # parsing the gxl
        # sets up the following properties: node_features, node_feature_names, edges, edge_features, edge_feature_names,
        # node_position, graph_id, edge_ids_present and edgemode
        self.setup_graph_features()

    @property
    def filepath(self) -> str:
        return self._filepath

    @filepath.setter
    def filepath(self, path_to_gxl):
        if not os.path.isfile(path_to_gxl):
            logging.error(f'File {path_to_gxl} does not exist.')
            sys.exit(-1)
        self._filepath = path_to_gxl

    @property
    def subset(self) -> str:
        return self._subset

    @subset.setter
    def subset(self, subset):
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

        """
        tree = ET.parse(self.filepath)
        root = tree.getroot()

        # verify that the file contains the expected attributes (node, edge, id, edgeids and edgemode)
        self.sanity_check(root)

        self.edges = self.get_edges(root)  # [[int, int]]
        self.node_feature_names, self.node_features = self.get_features(root, 'node')  # ([str], list)

        # remove the x and y node features and put them in their own variable
        if self.node_feature_names is not None and 'x' in self.node_feature_names and 'y' in self.node_feature_names:
            x_ind = self.node_feature_names.index('x')
            y_ind = self.node_feature_names.index('y')
            self.node_positions = [[node[x_ind], node[y_ind]] for node in self.node_features]
            # remove the positions from the node features if we don't want to use them
            if self.remove_coordinates:
                for node in self.node_features:
                    del node[x_ind]
                    del node[y_ind-1]
                del self.node_feature_names[x_ind]
                del self.node_feature_names[y_ind-1]
        else:
            self.node_positions = []  # [float / int]

        self.edge_feature_names, self.edge_features = self.get_features(root, 'edge')  # ([str], list)
        self.graph_id, self.edge_ids_present, self.edgemode = self.get_graph_attr(root)  # (str, bool, str)

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
            raise InvalidFileException
        elif len([edge for edge in root.iter('edge')]) == 0:
            logging.warning(f'File {self.filepath} has no edges!')

    def normalize(self, mean_std, center_coordinates):
        """
        # TODO implement this
        This method normalizes the node and edge features (if present) and initializes a random node for empty graphs

        Parameters
        ----------
        graph: ParsedGxlGraph

        mean_std: dict
            dictionary containing the mean and standard deviation for the node and edge features
        center_coordinates: set if coordinates need to be centered instead of normalized (xy-vector - xy(average)-vector)

        Returns
        ----------
        Normalized graph

        """
        # TODO: make this work for when only selected features are used
        def normalize(feature, mean, std, feature_name=None):
            # TODO: if coordinates, center otherwise do z-normalization
            if center_coordinates and feature_name in ['x', 'y', 'X', 'Y']:
                return feature - mean
            else:
                return (feature - mean) / std

        node_mean = mean_std['node_features']['mean']
        node_std = mean_std['node_features']['std']
        edge_mean = mean_std['edge_features']['mean']
        edge_std = mean_std['edge_features']['std']

        # check if graph is not empty
        if len(self.node_features) > 0:
            # normalize the node features
            for node_ind in range(len(self.node_features)):
                self.node_features[node_ind] = [
                    normalize(self.node_features[node_ind][i], node_mean[i], node_std[i], self.node_feature_names[i])
                    for i in range(len(node_mean))]
            # normalize the edge features (except the coordinates
            for edge_ind in range(len(self.edge_features)):
                self.edge_features[edge_ind] = [
                    normalize(self.edge_features[edge_ind][i], edge_mean[i], edge_std[i])
                    for i in range(len(edge_mean))]

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