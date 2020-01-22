import os
import xml.etree.ElementTree as ET
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from datasets.util.custom_exception import InvalidFileException


class ParsedGxlDataset:
    def __init__(self, path_to_dataset, categorical_features=None, subset="", disable_position=False,
                 features_to_use=None, no_empty_graphs=False):
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
        """
        self.no_empty_graphs = no_empty_graphs
        self.subset = subset
        self.disable_position = disable_position
        self.features_to_use = features_to_use
        if categorical_features is None:
            self.categorical_features = {'node': [], 'edge': []}
        else:
            self.categorical_features = categorical_features

        self.root_path = path_to_dataset
        self.name = os.path.basename(os.path.dirname(self.root_path))

        # read the train.cxl, valid.cxl and test.cxl and assign the graphs their class and split them into the sets
        self.dataset_split = self.get_dataset_split()

        self.invalid_files = []
        # if subset is specified only this is loaded, otherwise the whole dataset is loaded
        if len(self.subset) == 0:
            self.all_file_names = [f for f in os.listdir(path_to_dataset) if os.path.isfile(os.path.join(path_to_dataset, f)) if '.gxl' in f]
            self.graphs = self.get_graphs_all()
        else:
            assert self.subset in ['train', 'valid', 'test']
            self.all_file_names = [f for f in self.dataset_split[self.subset] if os.path.isfile(os.path.join(path_to_dataset, f)) if '.gxl' in f]
            self.graphs = self.get_graphs()
        # get a list of all the filenames
        self.file_names = [g.filename for g in self.graphs]

        # get the node and edge feature names available at a higher level
        agraph = [g for g in self.graphs if len(g.node_features) > 0][0]
        self.node_feature_names = agraph.node_feature_names
        self.edge_feature_names = agraph.edge_feature_names

        # node / edge feature names and data type
        if len(agraph.node_features) > 0:
            self.node_dtypes = [type(dtype) for dtype in agraph.node_features[0]]
        else:
            self.node_dtypes = None

        if self.node_feature_names is not None and self.node_dtypes is not None:
            assert len(self.node_feature_names) == len(self.node_dtypes)

        self.edge_dtypes = [type(dtype) for dtype in agraph.edge_features[0]] if self.edge_feature_names else None
        if self.edge_dtypes:
            assert len(self.edge_feature_names) == len(self.edge_dtypes)

        # if there are string-based features, we need to encode them as a one-hot vector
        if self.node_feature_names and (str in self.node_dtypes or len(self.categorical_features['node']) > 0):
            self.nodes_onehot = self.one_hot_encode_nodes()
        if self.edge_feature_names and (str in self.edge_dtypes or len(self.categorical_features['edge']) > 0):
            self.edges_onehot = self.one_hot_encode_edge_features()

        # setup the config dictionary
        self.config = self.config()

    def __len__(self):
        return len(self.graphs)

    @property
    def edge_mode(self):
        return self.graphs[0].edgemode

    def config(self):
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

    def get_dataset_split(self):
        """
        Create a dictionary that contains the dataset split as well as the class labels

        Returns
        -------
        filename_class_split: dict
            {'train': {'file1.gxl': 'class_label' }, ...}
        """
        filename_class_split = {}

        for subset in ['train', 'test', 'valid']:
            tree = ET.parse(os.path.join(self.root_path, subset + '.cxl'))
            root = tree.getroot()
            filename_class_split[subset] = {i.attrib['file']: i.attrib['class'] for i in root.iter('print')}

        return filename_class_split

    def one_hot_encode_nodes(self):
        """
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
            encoded_node_features += self.get_encoding(feature_name, all_feature_values, name_to_ind)

        # overwrite the categorical node features with the argmax of the one-hot encoding
        for graph_ind in range(self.__len__()):
            for node_ind in range(self.graphs[graph_ind].nb_of_nodes):
                for feature_ind, name, encoding in encoded_node_features:
                    self.graphs[graph_ind].node_features[node_ind][feature_ind] = np.argmax(encoding[
                        self.graphs[graph_ind].node_features[node_ind][feature_ind]])

        return encoded_node_features

    def one_hot_encode_edge_features(self):
        """
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
            encoded_edge_features += self.get_encoding(feature_name, all_feature_values, name_to_ind)

        # overwrite the categorical edge features with the argmax of the one-hot encoding
        for graph_ind in range(self.__len__()):
            for edge_ind in range(self.graphs[graph_ind].nb_of_edges):
                for feature_ind, name, encoding in encoded_edge_features:
                    self.graphs[graph_ind].edge_features[edge_ind][feature_ind] = np.argmax(encoding[
                        self.graphs[graph_ind].edge_features[edge_ind][feature_ind]])

        return encoded_edge_features

    def get_graphs_all(self):
        """

        Returns
        -------
        graphs: list [ParsedGxlGraph obj]
            list of graphs parsed from the gxl files
        """
        filename_split_class = {}
        for subset, d in self.dataset_split.items():
            for filename, class_label in d.items():
                filename_split_class[filename] = (subset, class_label)

        self.class_int_encoding = {c: i for i, c in enumerate(sorted(set([i[1] for i in filename_split_class.values()])))}
        self.classes = [c for c in sorted(set([i for i in filename_split_class.values()]))]

        graphs = []
        for filename in self.all_file_names:
            try:
                try:
                    subset, class_label = filename_split_class[filename]
                except KeyError:
                    print('{} does not appear in the dataset split files'.format(filename))

                g = ParsedGxlGraph(os.path.join(self.root_path, filename), subset, self.class_int_encoding[class_label],
                                   disable_position=self.disable_position, features_to_use=self.features_to_use, no_empty_graphs=self.no_empty_graphs)
                graphs.append(g)
            except InvalidFileException:
                print('File {} is invalid. Please verify that the file contains the expected attributes (id, edgeids, edgemode and nodes and edges, if expected)'.format(filename))
                self.invalid_files.append(filename)

        return graphs

    def get_graphs(self):
        """

        Returns
        -------
        graphs: list [ParsedGxlGraph obj]
            list of graphs parsed from the gxl files
        """
        filename_split_class = {}
        for filename, class_label in self.dataset_split[self.subset].items():
            filename_split_class[filename] = class_label

        self.class_int_encoding = {c: i for i, c in enumerate(sorted(set([i for i in filename_split_class.values()])))}
        self.classes = [c for c in sorted(set([i for i in filename_split_class.values()]))]

        graphs = []
        for filename in self.all_file_names:
            try:
                try:
                    class_label = filename_split_class[filename]
                except KeyError:
                    print('{} does not appear in the dataset split files'.format(filename))

                g = ParsedGxlGraph(os.path.join(self.root_path, filename), self.subset, self.class_int_encoding[class_label],
                                   disable_position=self.disable_position, features_to_use=self.features_to_use)
                graphs.append(g)
            except InvalidFileException:
                print('File {} is invalid. Please verify that the file contains the expected attributes (node, edge, id, edgeids and edgemode)'.format(filename))
                self.invalid_files.append(filename)

        return graphs

    @staticmethod
    def get_encoding(feature_name, all_feature_values, name_to_ind):
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


class ParsedGxlGraph:
    def __init__(self, path_to_gxl, subset, class_label, disable_position=False, features_to_use=None, no_empty_graphs=False):
        """
        This class contains all the information encoded in a single gxl file = one graph
        Parameters
        ----------
        subset : str
            either 'test', 'valid' or 'train'
        class_label : str
            class label of the graph
        path_to_gxl: str
            path to the gxl file
        """
        assert os.path.isfile(path_to_gxl)
        self.no_empty_graphs = no_empty_graphs

        self.features_to_use = features_to_use
        self.disable_position = disable_position
        # path to the gxl file
        self.filepath = path_to_gxl
        # name of the gxl file (without the ending)
        self.filename = os.path.basename(self.filepath).split('.')[0]
        # parsing the gxl to get the node and edge feature names and values
        graph_info, self.node_feature_names, self.node_features, self.node_position, self.edge_feature_names, \
            self.edge_features, self.edges = self.parse_gxl()
        # get the remaining info on the whole graph
        self.graph_id, self.edge_ids_present, self.edgemode = graph_info

        self.nb_of_nodes = len(self.node_features)
        self.nb_of_edges = len(self.edges)

        self.class_label = class_label
        self.subset = subset

    def parse_gxl(self):
        """
        Parses the gxl file and returns the graph info, node / feature names and values and the edge index list

        Returns
        -------
        self.get_graph_attr(root), node_feature_names, node_features, node_position, edge_feature_names, edge_features, edges:
            ( (str, bool, str), [str], [mixed], [float / int], [str], [mixed], [[int, int]] )
        """
        tree = ET.parse(self.filepath)
        root = tree.getroot()

        # verify that the file contains the expected attributes (node, edge, id, edgeids and edgemode)
        self.verification(root, self.no_empty_graphs)

        edges = self.get_edges(root)
        node_feature_names, node_features = self.get_features(root, 'node')

        # remove the x and y node features and put them in their own variable
        if node_feature_names is not None and 'x' in node_feature_names and 'y' in node_feature_names:
            x_ind = node_feature_names.index('x')
            y_ind = node_feature_names.index('y')
            node_position = [[node[x_ind], node[y_ind]] for node in node_features]
            # remove the positions from the node features if we don't want to use them
            if self.disable_position:
                for node in node_features:
                    del node[x_ind]
                    del node[y_ind-1]
                del node_feature_names[x_ind]
                del node_feature_names[y_ind-1]
        else:
            node_position = []

        edge_feature_names, edge_features = self.get_features(root, 'edge')
        #TODO: ensure that graph is undirected!
        return self.get_graph_attr(root), node_feature_names, node_features, node_position, edge_feature_names, edge_features, edges

    def get_features(self, root, mode):
        """
        get a list of the node features out of the element tree (gxl)

        Parameters
        ----------
        root: gxl element
        mode: str
            either 'edge' or 'node'

        Returns
        -------
        tuple ([str], [mixed values]])
            list of all node features for that tree
            ([[feature 1 of node 1, feature 2 of node 1, ...], [feature 1 of node 2, ...], ...])
        """
        features_info = [[feature for feature in node] for node in root.iter(mode)]
        if len(features_info) > 0:
            feature_names = [i.attrib['name'] for i in features_info[0]]
        else:
            feature_names = []

        # only use the ones specified, if necessary
        if self.features_to_use is not None:
            feature_names = [name for name in feature_names if name in self.features_to_use]

        # check if we have features to generate
        if len(feature_names) > 0:
            features = [[self.decode_feature(value) for feature in node for value in feature if feature.attrib['name'] in feature_names] for node in root.iter(mode)]
            # for debugging
            # features = []
            # for node in root.iter(mode):
            #     node_features = []
            #     for feature in node:
            #         if feature.attrib['name'] in feature_names:
            #             for value in feature:
            #                 node_features.append(self.decode_feature(value))
            #     features.append(node_features)
        else:
            feature_names = None
            features = []

        return feature_names, features


    @staticmethod
    def verification(root, no_empty_graphs):
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

        num_edges = len([edge for edge in root.iter('edge')])
        num_nodes = len([node for node in root.iter('node')])

        if no_empty_graphs:
            print('Excluding empty graphs')
            if len([edge for edge in root.iter('edge')]) == 0:
                raise InvalidFileException
            if len([node for node in root.iter('node')]) == 0:
                raise InvalidFileException

    @staticmethod
    def get_graph_attr(root):
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
    def get_edges(root):
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

        # move enumeration start to 0
        if len(start_points) > 0 and len(end_points) > 0:
            if min(min(start_points, end_points)) > 0:
                shift = min(min(start_points, end_points))
                start_points = [x - shift for x in start_points]
                end_points = [x - shift for x in end_points]

            edge_list = [[start_points[i], end_points[i]] for i in range(len(start_points))]

        return edge_list

    @staticmethod
    def decode_feature(f):
        data_types = {'string': str,
                      'float': float,
                      'int': int}

        # convert the feature value to the correct data type as specified in the gxl
        return data_types[f.tag](f.text.strip())


def normalize_graph(graph, mean_std):
    """
    This method normalizes the node and edge features (if present) and initializes a random node for empty graphs

    Parameters
    ----------
    graph: ParsedGxlGraph

    mean_std: dict
        dictionary containing the mean and standard deviation for the node and edge features

    Returns
    ----------
    Normalized graph

    """
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
            graph.node_features[node_ind] = [z_normalize(graph.node_features[node_ind][i], node_mean[i], node_std[i])
                                             for i in range(len(node_mean))]
        # normalize the edge features
        for edge_ind in range(len(graph.edge_features)):
            graph.edge_features[edge_ind] = [z_normalize(graph.edge_features[edge_ind][i], edge_mean[i], edge_std[i])
                                             for i in range(len(edge_mean))]

    return graph

